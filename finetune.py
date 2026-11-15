from ecg.config import CFG
from ecg.finetune import finetune_afib_afl
from ecg.utils import DEVICE, set_seed


def main():
    set_seed()
    print(f"[INFO] Device in use: {DEVICE}")
    finetune_afib_afl(CFG)
    print("\n[DONE] Fine-tuning completed.")


if __name__ == "__main__":
    main()
