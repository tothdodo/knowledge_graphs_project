"""Materialise the Datalog rules over the base graph."""

import argparse
from collections import Counter
import pandas as pd
import config
from datalog import DatalogEngine

# Load raw triples
def load_facts(triples_df: pd.DataFrame):
    for s, p, o in triples_df[["Subject", "Predicate", "Object"]].itertuples(index=False):
        yield p, (s, config.parse_object(o))

# Materialise the rules over the base graph
def materialise(base_df: pd.DataFrame, rules_path: str) -> DatalogEngine:
    engine = DatalogEngine()
    with open(rules_path, encoding="utf-8") as fh:
        engine.load_program(fh.read())
    engine.add_facts(load_facts(base_df))
    engine.evaluate()
    return engine


def main():
    parser = argparse.ArgumentParser(description="Apply Datalog rules to the base graph.")
    parser.add_argument("--base", default=config.BASE_GRAPH_CSV)
    parser.add_argument("--rules", default=config.RULES_FILE)
    parser.add_argument("--derived-out", default=config.DERIVED_CSV)
    parser.add_argument("--out", default=config.KG_CSV)
    args = parser.parse_args()

    base = pd.read_csv(args.base)
    engine = materialise(base, args.rules)
    derived = pd.DataFrame(engine.derived_triples(), columns=["Subject", "Predicate", "Object"])
    final = pd.concat([base, derived], ignore_index=True)

    derived.to_csv(args.derived_out, index=False)
    final.to_csv(args.out, index=False)

    print(f"Base triples: {len(base)}, derived triples: {len(derived)}, total: {len(final)} -> '{args.out}'")
    for pred, n in sorted(Counter(derived['Predicate']).items()):
        print(f"  {pred:24s} {n:5d}")
    with_arch = derived[derived["Predicate"] == "has_archetype"]["Subject"].nunique()
    players = base["Subject"].nunique()
    print(f"Players with >=1 archetype: {with_arch}/{players}. Upon that every player gets tier (6) + position-group (1) edges.")


if __name__ == "__main__":
    main()
