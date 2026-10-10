"""Predicts players with a similar playstyle (CLI)."""

import argparse
import config
from recommender import PlayerRecommender

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommend players with a similar playstyle.")
    parser.add_argument("--player", required=True, help="Player id ('doncilu01') or name ('Luka Doncic').")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--method", choices=["embedding", "rules"], default="embedding")
    parser.add_argument("--artifacts", default=config.ARTIFACTS_DIR)
    parser.add_argument("--data", default=config.KG_CSV)
    args = parser.parse_args()

    rec = PlayerRecommender(model_dir=args.artifacts, triples_csv_path=args.data)
    try:
        profile = rec.profile(args.player)
        results = rec.recommend(args.player, top_n=args.top_n, method=args.method)
    except (KeyError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}")

    print(f"\n{profile['name']} ({profile['player_id']}, {profile['team']}): {', '.join(profile['traits'])}")
    print(f"Top {len(results)} similar players [{args.method}]:")
    for r in results:
        print(f" {r['rank']}. {r['name']} ({r['player_id']}, {r['team']}, {r['pos']}) "
              f"score={r['score']}  {r['pts']} pts/{r['trb']} reb/{r['ast']} ast")
        print(f"    shares: {', '.join(r['shared_traits']) or '-'}")
