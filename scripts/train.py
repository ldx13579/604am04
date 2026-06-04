"""Train an offline RL algorithm from the command line."""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import Base, engine
from backend.algorithms.trainer import trainer


def main():
    parser = argparse.ArgumentParser(description="Train offline RL algorithm")
    parser.add_argument("--algorithm", choices=["cql", "dqn", "behavior_cloning"], default="cql")
    parser.add_argument("--alpha", type=float, default=1.0, help="CQL alpha (conservatism weight)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)

    hyperparams = {
        "alpha": args.alpha,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "steps_per_epoch": args.steps_per_epoch,
    }

    print(f"Training {args.algorithm} with params: {hyperparams}")
    run_id = trainer.start_training(args.algorithm, hyperparams)
    print(f"Training completed. Run ID: {run_id}")


if __name__ == "__main__":
    main()
