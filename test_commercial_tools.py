"""Report metrics for DALL-E 2, IF and Midjourney data."""

import argparse
from pathlib import Path

import numpy as np

from metrics import FIXED_THRESHOLD, MetricEvaluator, get_test_transform, make_loader, print_metrics, save_json
from fadnet import RealFakeDataset, image_files


DEFAULT_FAKE = {
    "dalle2": r"F:\CommercialTools\CommercialTools_CelebAHQ_processed\dalle2",
    "if": r"F:\CommercialTools\CommercialTools_CelebAHQ_processed\if",
    "midjourney": r"F:\CommercialTools\CommercialTools_CelebAHQ_processed\midjourney",
}
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "fadnet.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", default=r"E:\New folder3\archive\10000")
    parser.add_argument("--fake", action="append", metavar="NAME=PATH", help="Repeat for each synthetic source")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("results/commercial_tools.json"))
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    fake_items = dict(item.split("=", 1) for item in args.fake) if args.fake else DEFAULT_FAKE
    evaluator = MetricEvaluator(args.model)
    report = {"model": str(args.model), "threshold": FIXED_THRESHOLD, "datasets": {}}
    for name, fake_path in fake_items.items():
        fake_count = len(image_files(fake_path))
        if fake_count == 0:
            raise ValueError(f"No jpg/png images found in: {fake_path}")
        sample_limit = min(args.max_samples, fake_count)
        dataset = RealFakeDataset(args.real, fake_path, transform=get_test_transform(), max_samples=sample_limit)
        metrics = evaluator.evaluate(make_loader(dataset, args.batch_size), threshold=FIXED_THRESHOLD)
        print_metrics(name, metrics)
        report["datasets"][name] = metrics
    report["mean_metrics"] = {
        key: float(np.mean([item[key] for item in report["datasets"].values()]))
        for key in ("acc", "f1", "ap", "ece")
    }
    save_json(args.output, report)
    print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
