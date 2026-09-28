"""Fit and persist the digital-twin models for dashboard deployment."""

import argparse

from model_store import DEFAULT_MODEL_PATH, resolve_data_dir, train_and_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(resolve_data_dir()),
                        help="Directory containing the seven model input files")
    parser.add_argument("--output", default=str(DEFAULT_MODEL_PATH),
                        help="Destination .pkl model artifact")
    args = parser.parse_args()

    bundle = train_and_save(args.data_dir, args.output)
    print(f"Saved model artifact: {args.output}")
    print(f"Trained at: {bundle['trained_at']}")
    print(f"Training data: {bundle['training_data_dir']}")
    print(f"Dataset fingerprint: {bundle['training_data_fingerprint']}")
    print(f"Model counts: {bundle['training_summary']}")
    print(f"scikit-learn version: {bundle['versions']['scikit_learn']}")


if __name__ == "__main__":
    main()