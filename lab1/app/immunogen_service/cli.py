import argparse
import sys
from pathlib import Path

from .inference import ImmunogenicityPredictor, align_samples, results_to_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict antibody immunogenicity (ensemble AntiBERTy model)."
    )
    parser.add_argument("--heavy", required=True, type=Path, help="FASTA with heavy chains")
    parser.add_argument("--light", type=Path, help="Optional FASTA with light chains (matching IDs)")
    parser.add_argument("--model", type=Path, default=Path("/app/models/marks2021_ensemble.joblib"), help="Path to ensemble model")
    parser.add_argument("--out", type=Path, help="Output JSON file (default: stdout)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold on probability")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for embedding")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    samples = align_samples(args.heavy, args.light)
    predictor = ImmunogenicityPredictor(
        model_path=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )
    results = predictor.predict(samples, threshold=args.threshold)
    payload = results_to_json(results)

    if args.out:
        args.out.write_text(payload)
    else:
        sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
