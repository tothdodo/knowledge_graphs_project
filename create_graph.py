"""Build the base knowledge graph from raw CSV stats"""

import argparse
import pandas as pd
import config

COLUMNS_TO_EXCLUDE = {"Rk", "GS", "Awards", "Team"}

# Convert a column name to a predicate name
def predicate_name(column: str) -> str:
    return "has_" + column.replace("%", "_pct").replace("-", "_").replace(" ", "_")

# Build the base triples from CSV columns
def build_base_triples(df: pd.DataFrame) -> list[tuple]:
    triples = []
    for player_id, group in df.groupby("Player-additional", sort=False):
        if pd.isna(player_id):
            continue
        stats = group.iloc[0]
        
        # Filter out players with too few games or minutes
        if stats["G"] < config.MIN_GAMES or stats["MP"] < config.MIN_MINUTES:
            continue

        if len(group) > 1:
            teams = group["Team"].iloc[1:].dropna().unique()
        else:
            teams = [stats["Team"]] if not pd.isna(stats["Team"]) else []
        for team in teams:
            triples.append((player_id, config.TEAM_PREDICATE, team))

        for column in df.columns:
            if column in COLUMNS_TO_EXCLUDE:
                continue
            value = stats[column]
            if not pd.isna(value):
                triples.append((player_id, predicate_name(column), value))
    return triples


def main():
    parser = argparse.ArgumentParser(description="Build the base knowledge graph from raw stats.")
    parser.add_argument("--raw", default=config.RAW_CSV)
    parser.add_argument("--out", default=config.BASE_GRAPH_CSV)
    args = parser.parse_args()

    df = pd.read_csv(args.raw)
    triples = build_base_triples(df)
    out = pd.DataFrame(triples, columns=["Subject", "Predicate", "Object"])
    out.to_csv(args.out, index=False)
    print(f"Base graph: {len(out)} triples, {out['Subject'].nunique()} players -> '{args.out}'")


if __name__ == "__main__":
    main()
