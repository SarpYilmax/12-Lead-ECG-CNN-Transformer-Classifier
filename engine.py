import torch
import torch.nn as nn
from tqdm import tqdm

from ecg.metrics import compute_metrics


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device) -> dict:
    model.train()
    running_loss = 0.0
    all_labels, all_preds = [], []
    pbar = tqdm(loader, desc="  [Train]", leave=False, ncols=90)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item() * x.size(0)

        preds = logits.detach().argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []
    pbar = tqdm(loader, desc="  [Validation]", leave=False, ncols=90)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)

        preds = logits.detach().argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = running_loss / len(loader.dataset)
    metrics["all_labels"] = all_labels
    metrics["all_preds"] = all_preds
    return metrics


class EarlyStopping:
    def __init__(self, patience: int = 10, checkpoint_path: str = "best_model.pt"):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.best_f1 = -1.0
        self.counter = 0
        self.should_stop = False

    def step(self, val_f1: float, model: nn.Module) -> bool:
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"  [OK] New best model saved (Val Macro-F1={val_f1:.4f})")
        else:
            self.counter += 1
            print(f"  [--] No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
