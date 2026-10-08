"""Report metrics for all DF40 synthetic sources mixed together."""

import argparse
from pathlib import Path

from df40_config import DEFAULT_FAKE, DF40_ROOT
from metrics import FIXED_THRESHOLD, MetricEvaluator, get_test_transform, make_loader, print_metrics, save_json
from fadnet import MixedRealFakeDataset


DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "fadnet.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", default=r"E:\New folder3\archive\10000")
    parser.add_argument("--df40-root", type=Path, default=DF40_ROOT)
    parser.add_argument("--fake", action="append", metavar="NAME=PATH", help="Repeat to replace the default source list")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("results/df40_metrics.json"))
    parser.add_argument("--fake-max-samples", type=int, default=500)
    parser.add_argument("--real-max-samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    fake_items = list(dict(item.split("=", 1) for item in args.fake).items()) if args.fake else [(name, args.df40_root / relative) for name, relative in DEFAULT_FAKE.items()]
    dataset = MixedRealFakeDataset(args.real, fake_items, transform=get_test_transform(), real_max_samples=args.real_max_samples, fake_max_samples=args.fake_max_samples)
    evaluator = MetricEvaluator(args.model)
    metrics = evaluator.evaluate(make_loader(dataset, args.batch_size), threshold=FIXED_THRESHOLD)
    print_metrics("DF40 mixed", metrics)
    save_json(args.output, {"model": str(args.model), "threshold": FIXED_THRESHOLD, "sources": [name for name, _ in fake_items], "metrics": metrics})
    print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
