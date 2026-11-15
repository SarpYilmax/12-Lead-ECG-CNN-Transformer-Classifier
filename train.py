from ecg.config import CFG
from ecg.summary import print_model_summary
from ecg.training import train
from ecg.utils import DEVICE, set_seed


def main():
    set_seed()
    print(f"[INFO] Device in use: {DEVICE}")
    print_model_summary(CFG)
    train(CFG)
    print("\n[DONE] Training completed.")


if __name__ == "__main__":
    main()
