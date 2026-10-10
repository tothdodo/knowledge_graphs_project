"""
-Training: train the TransE model on the knowledge graph.
-Querying: find players similar to a given player based on their traits.
    -Two similarity methods:
        -embedding: cosine similarity of the TransE entity embeddings
        -rules: Jaccard overlap of the derived traits
"""

from __future__ import annotations
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import config

ENTITY_EMB = "entity_embeddings.npy"
RELATION_EMB = "relation_embeddings.npy"
ENTITY_IDS = "entity_to_id.json"
RELATION_IDS = "relation_to_id.json"
META = "meta.json"


# Training

def load_training_triples(triples_csv_path: str) -> pd.DataFrame:
    """Loads the triples CSV to train on."""
    df = pd.read_csv(triples_csv_path)
    predicates = list(config.PLAYSTYLE_PREDICATES)
    df = df[df["Predicate"].isin(predicates)]
    if df.empty:
        raise ValueError(f"No triples with predicates {predicates} found in '{triples_csv_path}'. "
                         f"Did you run create_graph.py and reason.py first?")
    return df[["Subject", "Predicate", "Object"]].astype(str).drop_duplicates()

def train_transe_model(
        triples_csv_path: str = config.KG_CSV,
        output_dir: str = config.ARTIFACTS_DIR,
        embedding_dim: int = 64,
        num_epochs: int = 200,
        batch_size: int = 256,
        random_seed: int = 42,
):
    """Train TransE on the triples of the knowledge graph."""
    import torch
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    df = load_training_triples(triples_csv_path)
    factory = TriplesFactory.from_labeled_triples(df.values)
    _, testing = factory.split([0.8, 0.2], random_state=random_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(train_tf, test_tf):
        return pipeline(
            training=train_tf,
            testing=test_tf,
            model="TransE",
            model_kwargs=dict(embedding_dim=embedding_dim),
            training_kwargs=dict(num_epochs=num_epochs, batch_size=batch_size),
            evaluation_kwargs=dict(batch_size=batch_size),
            random_seed=random_seed,
            device=device,
        )

    final = run(factory, testing)
    model = final.model

    # export embeddings + id mappings + metadata
    os.makedirs(output_dir, exist_ok=True)
    entity_emb = model.entity_representations[0](indices=None).detach().cpu().numpy()
    relation_emb = model.relation_representations[0](indices=None).detach().cpu().numpy()
    np.save(os.path.join(output_dir, ENTITY_EMB), entity_emb)
    np.save(os.path.join(output_dir, RELATION_EMB), relation_emb)
    _dump_json(os.path.join(output_dir, ENTITY_IDS), {k: int(v) for k, v in factory.entity_to_id.items()})
    _dump_json(os.path.join(output_dir, RELATION_IDS), {k: int(v) for k, v in factory.relation_to_id.items()})
    _dump_json(os.path.join(output_dir, META), {
        "model": "TransE", "embedding_dim": embedding_dim, "num_epochs": num_epochs,
        "batch_size": batch_size, "seed": random_seed,
        "num_triples": int(len(df)), "num_entities": int(factory.num_entities),
        "num_relations": int(factory.num_relations),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    try:    
        final.save_to_directory(output_dir)
    except Exception as exc:
        print(f"(note) could not save the PyKEEN model: {exc}")
    print(f"Embeddings and metadata saved to '{output_dir}'.")
    return final

def _dump_json(path, obj):
    """Write a JSON file"""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)

def _load_json(path):
    """Load a JSON file"""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# Querying

# Helper functions for normalising names and labelling traits
def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

# Tier labels
_TIER_LABEL = {"has_scoring_tier": "scoring", "has_playmaking_tier": "playmaking",
               "has_rebounding_tier": "rebounding", "has_three_point_tier": "3pt volume",
               "has_defensive_tier": "defensive activity", "has_role_tier": "role"}


# Trait label for a predicate/object pair
def trait_label(pred: str, obj: str) -> str:
    if pred == "has_archetype":
        return obj
    if pred == "has_archetype_family":
        return f"{obj} family"
    if pred in _TIER_LABEL:
        return f"{obj} {_TIER_LABEL[pred]}" if pred != "has_role_tier" else f"{obj} role"
    if pred in ("has_position_group", "has_Pos"):
        return obj
    return f"{pred}:{obj}"


class PlayerRecommender:
    """Loads exported embeddings + KG and handles queryies like: 'players similar to X'."""

    def __init__(self, model_dir: str = config.ARTIFACTS_DIR, triples_csv_path: str = config.KG_CSV,
                 center: bool = True):
        self.center = center
        self._load_graph(triples_csv_path)
        self._load_embeddings(model_dir)
        self._build_player_matrices()

    # Loading the graph
    def _load_graph(self, path: str):
        df = pd.read_csv(path)
        self.names, self.teams, self.stats, self.traits = {}, {}, {}, {}
        explain = set(config.EXPLAIN_PREDICATES) | {"has_Pos"}
        for s, p, o in df[["Subject", "Predicate", "Object"]].itertuples(index=False):
            if p == "has_Player":
                self.names[s] = str(o)
            elif p == config.TEAM_PREDICATE:
                self.teams.setdefault(s, []).append(str(o))
            elif p in explain:
                self.traits.setdefault(s, set()).add((p, str(o)))
            elif p.startswith("has_") and p not in ("has_Player_additional",):
                self.stats.setdefault(s, {})[p[4:]] = config.parse_object(o)
        # players = subjects that have a name and derived traits
        self.player_ids = sorted(s for s in self.names if s in self.traits)
        self._name_index = {}
        for pid in self.player_ids:
            self._name_index.setdefault(_normalise(self.names[pid]), []).append(pid)

    # Loading the embeddings
    def _load_embeddings(self, model_dir: str):
        if not os.path.exists(os.path.join(model_dir, ENTITY_EMB)):
            self.entity_emb, self.entity_to_id, self.relation_emb, self.relation_to_id = None, {}, None, {}
            self.meta = {}
            return
        self.entity_emb = np.load(os.path.join(model_dir, ENTITY_EMB))
        self.relation_emb = np.load(os.path.join(model_dir, RELATION_EMB)) \
            if os.path.exists(os.path.join(model_dir, RELATION_EMB)) else None
        self.entity_to_id = _load_json(os.path.join(model_dir, ENTITY_IDS))
        rel_path = os.path.join(model_dir, RELATION_IDS)
        self.relation_to_id = _load_json(rel_path) if os.path.exists(rel_path) else {}
        meta_path = os.path.join(model_dir, META)
        self.meta = _load_json(meta_path) if os.path.exists(meta_path) else {}

    # Build two easily accessible matrices for the two similarity methods
    def _build_player_matrices(self):
        # Trait matrix for the logical Jaccard similarity (Rules)
        # Easily accesible for all players
        vocab = sorted({t for pid in self.player_ids for t in self.traits[pid]})
        col = {t: i for i, t in enumerate(vocab)}
        self.trait_matrix = np.zeros((len(self.player_ids), len(vocab)), dtype=np.float32)
        for r, pid in enumerate(self.player_ids):
            for t in self.traits[pid]:
                self.trait_matrix[r, col[t]] = 1.0
        # Embedding matrix for the cosine similarity (TransE)
        # Easily accesible for all players
        self.embedded_ids = [p for p in self.player_ids if p in self.entity_to_id]
        if self.entity_emb is not None and self.embedded_ids:
            mat = self.entity_emb[[self.entity_to_id[p] for p in self.embedded_ids]].astype(np.float64)
            if self.center:
                mat = mat - mat.mean(axis=0, keepdims=True)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            self._emb_unit = mat / np.maximum(norms, 1e-12)
        else:
            self._emb_unit = None

    # Player lookup
    def resolve(self, query: str) -> str:
        """Accepts a player id ('doncilu01') or a (partial) name ('luka doncic')."""
        if query in self.names:
            return query
        q = _normalise(query)
        if q in self._name_index and len(self._name_index[q]) == 1:
            return self._name_index[q][0]
        hits = [pid for pid in self.player_ids if q and q in _normalise(self.names[pid])]
        if len(hits) == 1:
            return hits[0]
        if hits:
            options = ", ".join(f"{self.names[h]} ({h})" for h in hits[:8])
            raise KeyError(f"'{query}' is ambiguous. Did you mean: {options}?")
        raise KeyError(f"Player '{query}' is not present in the Knowledge Graph.")

    # Search by (partial) name
    def search(self, text: str, limit: int = 10) -> list[dict]:
        q = _normalise(text)
        hits = [pid for pid in self.player_ids if q in _normalise(self.names[pid]) or q in pid.lower()]
        return [{"player_id": p, "name": self.names[p]} for p in hits[:limit]]

    # Similarity scores based on 'embedding' or 'rules' method
    def _scores(self, target_id: str, method: str) -> tuple[list[str], np.ndarray]:
        if method == "embedding":
            if self._emb_unit is None:
                raise RuntimeError("No embeddings loaded - run train.py first (or use method='rules').")
            if target_id not in self.embedded_ids:
                raise KeyError(f"Player '{target_id}' has no trained embedding.")
            t = self.embedded_ids.index(target_id)
            return self.embedded_ids, self._emb_unit @ self._emb_unit[t]
        if method == "rules":
            t = self.player_ids.index(target_id)
            inter = self.trait_matrix @ self.trait_matrix[t]
            union = self.trait_matrix.sum(1) + self.trait_matrix[t].sum() - inter
            return self.player_ids, inter / np.maximum(union, 1e-12)
        raise ValueError("method must be 'embedding' or 'rules'")

    # Predict the most similar players to a target player
    def predict(self, target_player_id: str, top_n: int = 5, method: str = "embedding") -> list[tuple[str, float]]:
        return [(r["player_id"], r["score"]) for r in self.recommend(target_player_id, top_n, method)]

    # Get the top most similar N players to a target player
    def recommend(self, target: str, top_n: int = 5, method: str = "embedding") -> list[dict]:
        target_id = self.resolve(target)
        ids, scores = self._scores(target_id, method)
        order = np.argsort(-scores, kind="stable")
        results = []
        target_traits = self.traits[target_id]
        for idx in order:
            pid = ids[idx]
            if pid == target_id:
                continue
            shared = sorted(trait_label(p, o) for p, o in (self.traits[pid] & target_traits)
                            if p not in ("has_Pos", "has_archetype_family"))
            results.append({
                "rank": len(results) + 1,
                "player_id": pid,
                "name": self.names[pid],
                "team": "/".join(self.teams.get(pid, [])) or None,
                "pos": next((o for p, o in self.traits[pid] if p == "has_Pos"), None),
                "score": round(float(scores[idx]), 4),
                "pts": self.stats.get(pid, {}).get("PTS"),
                "trb": self.stats.get(pid, {}).get("TRB"),
                "ast": self.stats.get(pid, {}).get("AST"),
                "shared_traits": shared,
            })
            if len(results) >= top_n:
                break
        return results

    # Get a player's profile
    def profile(self, player: str) -> dict:
        pid = self.resolve(player)
        return {"player_id": pid, "name": self.names[pid], "team": "/".join(self.teams.get(pid, [])) or None,
                "traits": sorted(trait_label(p, o) for p, o in self.traits[pid] if p != "has_archetype_family"),
                "stats": {k: self.stats[pid].get(k) for k in ("PTS", "TRB", "AST", "STL", "BLK", "3P", "3PA")}}
