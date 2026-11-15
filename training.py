import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

from ecg.data.builders import build_combined_dataframes, make_loader
from ecg.data.dataset import ECGDataset
from ecg.engine import EarlyStopping, evaluate, train_one_epoch
from ecg.losses import HybridLoss
from ecg.models import ECGHybridClassifier
from ecg.utils import DEVICE


def train(cfg: dict):
    print("\n" + "=" * 70)
    print("  ECG Hybrid CNN+Transformer - Merging datasets...")
    print("=" * 70)

    df_train_combined, df_val_combined = build_combined_dataframes(cfg)

    train_ds = ECGDataset(df_train_combined, cfg["data_root"], cfg["seq_len"], augment=True)
    val_ds = ECGDataset(df_val_combined, cfg["data_root"], cfg["seq_len"], augment=False)

    class_counts = df_train_combined["class_id"].value_counts().sort_index()
    class_weights = 1.0 / class_counts
    sample_weights = df_train_combined["class_id"].map(class_weights).values
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = make_loader(train_ds, cfg, sampler=sampler, drop_last=True)
    val_loader = make_loader(val_ds, cfg)

    print(f"  Combined training set   : {len(train_ds):,} records")
    print(f"  Combined validation set : {len(val_ds):,} records")
    print("  [INFO] WeightedRandomSampler enabled. Training class distribution:")
    for cid, cnt in class_counts.items():
        print(f"    - {cfg['class_names'][cid]}: {cnt:,} records")

    model = ECGHybridClassifier(cfg).to(DEVICE)

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        print(f"  [INFO] {n_gpu} GPUs found, enabling DataParallel.")
        model = nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    criterion = HybridLoss(
        focal_weight=0.7,
        ce_weight=0.3,
        gamma=cfg["focal_gamma"],
        alpha=cfg["focal_alpha"],
        label_smoothing=cfg["label_smoothing"],
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["lr"],
        steps_per_epoch=len(train_loader),
        epochs=cfg["epochs"],
        pct_start=0.1,
        anneal_strategy="cos",
    )
    early_stop = EarlyStopping(cfg["patience"], cfg["checkpoint_path"])
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        print(f"\nEpoch {epoch:>3}/{cfg['epochs']}  (LR={scheduler.get_last_lr()[0]:.2e})")
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, DEVICE)
        val_m = evaluate(model, val_loader, criterion, DEVICE)

        history.append({
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_acc": train_m["accuracy"],
            "train_b_acc": train_m["balanced_accuracy"],
            "train_macro_f1": train_m["macro_f1"],
            "train_weighted_f1": train_m["weighted_f1"],
            "val_loss": val_m["loss"],
            "val_acc": val_m["accuracy"],
            "val_b_acc": val_m["balanced_accuracy"],
            "val_macro_f1": val_m["macro_f1"],
            "val_weighted_f1": val_m["weighted_f1"],
        })

        print(f"  Train -> Loss:{train_m['loss']:.4f} | Acc:{train_m['accuracy']:.4f} | "
              f"Bal.Acc:{train_m['balanced_accuracy']:.4f} | Macro F1:{train_m['macro_f1']:.4f}")
        print(f"  Val   -> Loss:{val_m['loss']:.4f} | Acc:{val_m['accuracy']:.4f} | "
              f"Bal.Acc:{val_m['balanced_accuracy']:.4f} | Macro F1:{val_m['macro_f1']:.4f}")

        if early_stop.step(val_m["macro_f1"], model):
            print(f"\n[INFO] Early stopping triggered. Stopped at epoch {epoch}.")
            break

    print("\n" + "-" * 70)
    print(f"  Training finished. Best Val Macro-F1: {early_stop.best_f1:.4f}")
    pd.DataFrame(history).to_csv("training_history.csv", index=False)
    return model, history
