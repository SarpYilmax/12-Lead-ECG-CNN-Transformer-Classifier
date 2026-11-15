from pathlib import Path

from ecg.config import CFG
from ecg.finetune import finetune_afib_afl
from ecg.summary import print_model_summary
from ecg.testing import test_evaluate
from ecg.training import train
from ecg.utils import DEVICE, set_seed


def main():
    set_seed()
    print(f"[INFO] Device in use: {DEVICE}")

    print_model_summary(CFG)
    train(CFG)

    if not Path(CFG["test_csv"]).exists():
        print(f"\n[WARNING] '{CFG['test_csv']}' not found, skipping evaluation.")
        print("\n[DONE] All steps completed.")
        return

    print("\n\n########## ORIGINAL MODEL TEST RESULTS ##########")
    test_evaluate(CFG, checkpoint_path=CFG["checkpoint_path"], run_saliency=True)

    finetune_afib_afl(CFG)

    print("\n\n########## FINE-TUNED MODEL (AFIB/AFL FOCUSED) TEST RESULTS ##########")
    test_evaluate(CFG, checkpoint_path=CFG["finetune_checkpoint_path"], run_saliency=False)

    print("\n" + "=" * 80)
    print("  COMPARISON COMPLETE")
    print("  Compare the two overall-metrics and per-class blocks above:")
    print("  check which model has the higher Macro-F1 and that Normal/LBBB/RBBB")
    print("  have not degraded. Use the better one for the final submission.")
    print("=" * 80)

    print("\n[DONE] All steps completed.")


if __name__ == "__main__":
    main()
