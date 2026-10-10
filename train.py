"""Train TransE on the playstyle part of the knowledge graph"""

import argparse
import config
from recommender import train_transe_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TransE on the basketball Knowledge Graph.")
    parser.add_argument("--data", default=config.KG_CSV, help="Path to the triples CSV (base + derived).")
    parser.add_argument("--output", default=config.ARTIFACTS_DIR, help="Directory for the exported artifacts.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_transe_model(
        triples_csv_path=args.data,
        output_dir=args.output,
        embedding_dim=args.dim,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        random_seed=args.seed,
    )
