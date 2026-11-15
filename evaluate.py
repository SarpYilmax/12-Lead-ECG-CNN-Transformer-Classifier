import argparse
from pathlib import Path

from ecg.config import CFG
from ecg.testing import test_evaluate
from ecg.utils import DEVICE, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an ECG checkpoint on the test set.")
    parser.add_argument("--checkpoint", type=str, default=CFG["checkpoint_path"])
    parser.add_argument("--no-saliency", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed()
    print(f"[INFO] Device in use: {DEVICE}")

    if not Path(CFG["test_csv"]).exists():
        print(f"[WARNING] '{CFG['test_csv']}' not found, skipping evaluation.")
        return

    test_evaluate(CFG, checkpoint_path=args.checkpoint, run_saliency=not args.no_saliency)
    print("\n[DONE] Evaluation completed.")


if __name__ == "__main__":
    main()
