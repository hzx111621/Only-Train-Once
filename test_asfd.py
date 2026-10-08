"""Report metrics for the ASFD generator datasets."""

import argparse
from pathlib import Path

from metrics import FIXED_THRESHOLD, MetricEvaluator, get_test_transform, make_loader, print_metrics, save_json
from fadnet import RealFakeDataset


DEFAULT_FAKE = {
    "ProGAN": r"E:\New folder3\gan\ProGAN",
    "StyleGAN": r"E:\New folder3\gan\StyleGAN",
    "StyleGAN2": r"E:\New folder3\gan\StyleGAN2-20251016T055258Z-1-001\StyleGAN2",
    "VQGAN": r"E:\New folder3\gan\VQGAN-20251016T055154Z-1-001\VQGAN",
    "ADM": r"E:\New folder3\dm\ADM-20251016T055152Z-1-001\ADM",
    "IDDPM": r"E:\New folder3\dm\IDDPM-20251016T055256Z-1-001\IDDPM",
    "LDM": r"E:\New folder3\dm\LDM",
}
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "fadnet.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", default=r"E:\New folder3\archive\10000")
    parser.add_argument("--fake", action="append", metavar="NAME=PATH", help="Repeat for each synthetic source")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("results/asfd_metrics.json"))
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    threshold_mode = parser.add_mutually_exclusive_group()
    threshold_mode.add_argument("--find-best-threshold", dest="find_best_threshold", action="store_true")
    threshold_mode.add_argument("--fixed-threshold", dest="find_best_threshold", action="store_false")
    parser.set_defaults(find_best_threshold=True)
    args = parser.parse_args()
    fake_items = dict(item.split("=", 1) for item in args.fake) if args.fake else DEFAULT_FAKE
    evaluator = MetricEvaluator(args.model)
    report = {"model": str(args.model), "datasets": {}}
    for name, fake_path in fake_items.items():
        dataset = RealFakeDataset(args.real, fake_path, transform=get_test_transform(), max_samples=args.max_samples)
        loader = make_loader(dataset, args.batch_size)
        threshold = evaluator.find_best_threshold(loader) if args.find_best_threshold else FIXED_THRESHOLD
        metrics = evaluator.evaluate(loader, threshold=threshold)
        print_metrics(name, metrics)
        report["datasets"][name] = metrics
    save_json(args.output, report)
    print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
