"""
Evolve the knowledge graph without rebuilding it.

Two incremental change is possible:
    1. New logical knowledge: New Datalog rules as .dl file
    2. New data: New players in a CSV file
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import config
import recommender as rc
from create_graph import build_base_triples
from datalog import DatalogEngine
from reason import load_facts

# Helper read function
def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# Embeddings for new entities
def extend_embeddings(new_triples, entity_emb, relation_emb, entity_to_id, relation_to_id):
    emb = {e: entity_emb[i] for e, i in entity_to_id.items()}
    usable = [(s, p, str(o)) for s, p, o in new_triples if p in relation_to_id]
    added = []

    # Pass unknown tails (new classes)
    tails = {}
    for s, p, o in usable:
        if o not in emb and s in emb:
            tails.setdefault(o, []).append(emb[s] + relation_emb[relation_to_id[p]])
    for o, vecs in tails.items():
        emb[o] = np.mean(vecs, axis=0)
        added.append(o)

    # Pass unknown heads (new players)
    heads = {}
    for s, p, o in usable:
        if s not in emb and o in emb:
            heads.setdefault(s, []).append(emb[o] - relation_emb[relation_to_id[p]])
    for s, vecs in heads.items():
        emb[s] = np.mean(vecs, axis=0)
        added.append(s)

    new_map = dict(entity_to_id)
    rows = [entity_emb]
    for e in added:
        new_map[e] = len(new_map)
        rows.append(emb[e][None, :])
    return np.vstack(rows), new_map, added

# Check of the inductive embedding: rebuild the embedding of existing players
# and check how highly the trained embedding of the same player ranks.
def inductive_self_test(kg_df, entity_emb, relation_emb, entity_to_id, relation_to_id, sample=100, seed=0):
    train_preds = set(relation_to_id)
    sub = kg_df[kg_df["Predicate"].isin(train_preds)]
    players = [p for p in sub["Subject"].unique() if p in entity_to_id]
    rng = np.random.default_rng(seed)
    players = list(rng.choice(players, size=min(sample, len(players)), replace=False))
    pool = np.array([entity_emb[entity_to_id[p]] for p in players])
    pool = pool - pool.mean(0)
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    mean = np.mean([entity_emb[entity_to_id[p]] for p in players], axis=0)
    ranks = []
    for i, p in enumerate(players):
        rows = sub[sub["Subject"] == p]
        vecs = [entity_emb[entity_to_id[str(o)]] - relation_emb[relation_to_id[pr]]
                for pr, o in zip(rows["Predicate"], rows["Object"]) if str(o) in entity_to_id]
        v = np.mean(vecs, axis=0) - mean
        v /= max(np.linalg.norm(v), 1e-12)
        sims = pool @ v
        ranks.append(int((sims > sims[i]).sum()) + 1)
    ranks = np.array(ranks)
    return {"n": len(players), "hits@1": float((ranks == 1).mean()), "hits@5": float((ranks <= 5).mean()),
            "mean_rank": float(ranks.mean())}


# Evolution step
def evolve(kg_csv, rules_path, artifacts_dir, out_kg, out_artifacts, new_rules_path=None, new_players_csv=None,
           verify=False):
    log = {}
    kg = pd.read_csv(kg_csv)

    # Restore the reasoner state: rules of prev version + all existing triples as facts
    engine = DatalogEngine()
    engine.load_program(_read(rules_path))
    engine.add_facts(load_facts(kg))
    engine.evaluate()

    new_triples: list[tuple] = []

    # If there are new rules, compute their consequences only (incremental reasoning)
    if new_rules_path:
        text = _read(new_rules_path)
        new_rules, delta = engine.add_rules(text)
        t0 = time.perf_counter()
        inferred = engine.evaluate(delta, first_round_rules=new_rules)
        log["new_rules"] = {"file": new_rules_path, "rules_added": len(new_rules), "facts_inferred": inferred,
                            "seconds": round(time.perf_counter() - t0, 4)}
        new_triples += engine.derived_triples()
        seen = set(new_triples)
    else:
        seen = set()

    # If there are new players, add them and compute their consequences only (incremental reasoning)
    if new_players_csv:
        df = pd.read_csv(new_players_csv)
        base = build_base_triples(df)
        base_df = pd.DataFrame(base, columns=["Subject", "Predicate", "Object"])
        t0 = time.perf_counter()
        delta = engine.add_facts(load_facts(base_df))
        inferred = engine.evaluate(delta)
        log["new_players"] = {"file": new_players_csv, "players": sorted(base_df["Subject"].unique()),
                              "base_triples": len(base), "facts_inferred": inferred,
                              "seconds": round(time.perf_counter() - t0, 4)}
        new_triples += base
        new_triples += [t for t in engine.derived_triples() if t not in seen]

    new_df = pd.DataFrame(new_triples, columns=["Subject", "Predicate", "Object"]).drop_duplicates()
    out = pd.concat([kg, new_df], ignore_index=True).drop_duplicates()
    out.to_csv(out_kg, index=False)
    log["kg"] = {"triples_before": len(kg), "triples_added": len(out) - len(kg), "triples_after": len(out),
                 "added_by_predicate": new_df["Predicate"].value_counts().to_dict()}

    # Verify incremental reasoning against full recomputation (we didnt ruin anything or made duplicates)
    if verify:
        full = DatalogEngine()
        full.load_program(_read(rules_path))
        if new_rules_path:
            full.load_program(_read(new_rules_path))
        base_only = kg[~kg["Predicate"].isin(full.outputs)]
        extra = pd.DataFrame(build_base_triples(pd.read_csv(new_players_csv)), columns=kg.columns) \
            if new_players_csv else kg.iloc[0:0]
        full.add_facts(load_facts(pd.concat([base_only, extra])))
        full.evaluate()
        scratch = {(s, p, str(o)) for s, p, o in full.derived_triples()}
        incremental = {(s, p, str(o)) for s, p, o in out.itertuples(index=False) if p in full.outputs}
        log["verified_equal_to_full_recomputation"] = scratch == incremental

    # Embeddings for the new entities and relations
    ent_path = os.path.join(artifacts_dir, rc.ENTITY_EMB)
    if os.path.exists(ent_path):
        E = np.load(ent_path)
        R = np.load(os.path.join(artifacts_dir, rc.RELATION_EMB))
        e2i = rc._load_json(os.path.join(artifacts_dir, rc.ENTITY_IDS))
        r2i = rc._load_json(os.path.join(artifacts_dir, rc.RELATION_IDS))
        log["inductive_self_test_on_existing_players"] = inductive_self_test(kg, E, R, e2i, r2i)
        E2, e2i2, added = extend_embeddings(list(new_df.itertuples(index=False, name=None)), E, R, e2i, r2i)
        os.makedirs(out_artifacts, exist_ok=True)
        np.save(os.path.join(out_artifacts, rc.ENTITY_EMB), E2)
        np.save(os.path.join(out_artifacts, rc.RELATION_EMB), R)
        rc._dump_json(os.path.join(out_artifacts, rc.ENTITY_IDS), e2i2)
        rc._dump_json(os.path.join(out_artifacts, rc.RELATION_IDS), r2i)
        meta_path = os.path.join(artifacts_dir, rc.META)
        meta = rc._load_json(meta_path) if os.path.exists(meta_path) else {}
        meta.update({"evolved_from": artifacts_dir, "entities_added_inductively": added})
        rc._dump_json(os.path.join(out_artifacts, rc.META), meta)
        log["embeddings"] = {"new_entities_embedded": added}
    else:
        log["embeddings"] = "skipped (no trained artifacts found - run train.py first)"
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incrementally evolve the KG (new rules and/or new players).")
    parser.add_argument("--kg", default=config.KG_CSV)
    parser.add_argument("--rules", default=config.RULES_FILE)
    parser.add_argument("--artifacts", default=config.ARTIFACTS_DIR)
    parser.add_argument("--new-rules", default=None)
    parser.add_argument("--new-players", default=None)
    parser.add_argument("--out-kg", default="data/final_knowledge_graph_v2.csv")
    parser.add_argument("--out-artifacts", default="artifacts_v2")
    parser.add_argument("--log", default="results/evolution_log.json")
    parser.add_argument("--verify", action="store_true", help="check incremental result against full recomputation")
    args = parser.parse_args()

    log = evolve(args.kg, args.rules, args.artifacts, args.out_kg, args.out_artifacts,
                 args.new_rules, args.new_players, args.verify)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, ensure_ascii=False)
    print(json.dumps(log, indent=2, ensure_ascii=False))