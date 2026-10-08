"""Report metrics for one real/synthetic dataset pair."""

import argparse
from pathlib import Path

from metrics import FIXED_THRESHOLD, MetricEvaluator, get_test_transform, make_loader, print_metrics, save_json
from fadnet import RealFakeDataset


DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "fadnet.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True)
    parser.add_argument("--fake", required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("results/test_metrics.json"))
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=FIXED_THRESHOLD)
    args = parser.parse_args()
    dataset = RealFakeDataset(args.real, args.fake, transform=get_test_transform(), max_samples=args.max_samples)
    metrics = MetricEvaluator(args.model).evaluate(make_loader(dataset, args.batch_size), threshold=args.threshold)
    print_metrics("dataset", metrics)
    save_json(args.output, {"model": str(args.model), "real": args.real, "fake": args.fake, "metrics": metrics})
    print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
