import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support)
from tqdm import tqdm

from ecg.checkpoint import load_checkpoint
from ecg.data.builders import make_loader
from ecg.data.dataset import ECGDataset
from ecg.models import ECGHybridClassifier
from ecg.saliency import generate_saliency_maps
from ecg.utils import DEVICE


def test_evaluate(cfg: dict, checkpoint_path: str = None, run_saliency: bool = True):
    if checkpoint_path is None:
        checkpoint_path = cfg["checkpoint_path"]

    print("\n" + "=" * 80)
    print(f"  TEST SET EVALUATION ({os.path.basename(checkpoint_path)})")
    print("=" * 80)

    df_test = pd.read_csv(cfg["test_csv"])
    df_test["is_physionet"] = False

    test_ds = ECGDataset(df_test, cfg["data_root"], cfg["seq_len"], augment=False)
    test_loader = make_loader(test_ds, cfg)

    model = ECGHybridClassifier(cfg).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    load_checkpoint(model, checkpoint_path, DEVICE)
    model.eval()

    all_labels, all_preds = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="  Computing test metrics", ncols=90):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    b_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print("\n  [OVERALL METRICS]")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Balanced Accuracy : {b_acc:.4f}")
    print(f"  Macro F1 Score    : {macro_f1:.4f}")

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(cfg["num_classes"]), zero_division=0
    )

    print("\n  [PER-CLASS PERFORMANCE REPORT]")
    print(f"  {'-' * 80}")
    print(f"  {'Class':<12} | {'Precision':<16} | {'Recall':<16} | {'F1 Score':<10} | {'Support':<10}")
    print(f"  {'-' * 80}")
    for i, class_name in enumerate(cfg["class_names"]):
        print(f"  {class_name:<12} | {precisions[i]:<16.4f} | {recalls[i]:<16.4f} | "
              f"{f1s[i]:<10.4f} | {supports[i]:<10}")
    print(f"  {'-' * 80}\n")

    print("  [INFO] Creating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cfg["class_names"])

    tag = os.path.splitext(os.path.basename(checkpoint_path))[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Test Set - Confusion Matrix ({tag})")
    plt.tight_layout()
    cm_path = f"confusion_matrix_{tag}.png"
    plt.savefig(cm_path, dpi=300)
    plt.show()
    print(f"  [OK] Confusion matrix saved as '{cm_path}'.\n")

    if run_saliency:
        generate_saliency_maps(model, test_loader, cfg, tag)
