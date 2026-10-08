"""Report metrics separately for each DF40 synthetic source."""

import argparse
from pathlib import Path

import numpy as np

from df40_config import DEFAULT_FAKE, DF40_ROOT
from metrics import FIXED_THRESHOLD, MetricEvaluator, get_test_transform, make_loader, print_metrics, save_json
from fadnet import RealFakeDataset


DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "fadnet.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", default=r"E:\New folder3\archive\10000")
    parser.add_argument("--df40-root", type=Path, default=DF40_ROOT)
    parser.add_argument("--fake", action="append", metavar="NAME=PATH", help="Repeat to replace the default source list")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("results/df40_per_source.json"))
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    fake_items = dict(item.split("=", 1) for item in args.fake) if args.fake else {name: args.df40_root / relative for name, relative in DEFAULT_FAKE.items()}
    evaluator = MetricEvaluator(args.model)
    transform = get_test_transform()
    report = {"model": str(args.model), "threshold": FIXED_THRESHOLD, "datasets": {}}
    for name, fake_path in fake_items.items():
        dataset = RealFakeDataset(args.real, fake_path, transform=transform, max_samples=args.max_samples)
        metrics = evaluator.evaluate(make_loader(dataset, args.batch_size), threshold=FIXED_THRESHOLD)
        print_metrics(name, metrics)
        report["datasets"][name] = metrics
    keys = ("acc", "f1", "ap", "ece")
    report["mean_metrics"] = {
        key: float(np.mean([item[key] for item in report["datasets"].values()]))
        for key in keys
    }
    save_json(args.output, report)
    print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
