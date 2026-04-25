import os
import random
from typing import Dict, List
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from cnn import CNN
from data_pipeline import build_dataloaders


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> Dict[str, object]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    batch_losses: List[float] = []
    batch_accs: List[float] = []

    pbar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        preds = logits.argmax(dim=1)
        running_loss += loss.item() * batch_size
        running_correct += (preds == labels).sum().item()
        running_total += batch_size
        batch_acc = (preds == labels).float().mean().item()
        batch_losses.append(loss.item())
        batch_accs.append(batch_acc)

        pbar.set_postfix(
            loss=f"{running_loss / running_total:.4f}",
            acc=f"{running_correct / running_total:.4f}",
        )

    return {
        "loss": running_loss / running_total,
        "acc": running_correct / running_total,
        "batch_losses": batch_losses,
        "batch_accs": batch_accs,
    }


@torch.no_grad()
def infer_with_metrics(
    model: nn.Module,
    test_loader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    epochs: int,
    num_classes: int = 10,
) -> Dict[str, object]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    batch_losses: List[float] = []
    batch_accs: List[float] = []
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.long)

    pbar = tqdm(test_loader, desc=f"Infer {epoch}/{epochs}", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (preds == labels).sum().item()
        running_total += batch_size
        batch_acc = (preds == labels).float().mean().item()
        batch_losses.append(loss.item())
        batch_accs.append(batch_acc)

        for t, p in zip(labels.cpu(), preds.cpu()):
            conf_mat[t.long(), p.long()] += 1

        pbar.set_postfix(
            loss=f"{running_loss / running_total:.4f}",
            acc=f"{running_correct / running_total:.4f}",
        )

    tp = conf_mat.diag().float()
    fp = conf_mat.sum(dim=0).float() - tp
    fn = conf_mat.sum(dim=1).float() - tp
    eps = 1e-12

    per_class_precision = tp / (tp + fp + eps)
    per_class_recall = tp / (tp + fn + eps)
    per_class_f1 = (
        2
        * per_class_precision
        * per_class_recall
        / (per_class_precision + per_class_recall + eps)
    )

    precision_macro = per_class_precision.mean().item()
    recall_macro = per_class_recall.mean().item()
    f1_macro = per_class_f1.mean().item()

    return {
        "loss": running_loss / running_total,
        "acc": running_correct / running_total,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "batch_losses": batch_losses,
        "batch_accs": batch_accs,
    }


def plot_batch_curves(
    train_batch_losses: List[float],
    train_batch_accs: List[float],
    infer_batch_losses: List[float],
    infer_batch_accs: List[float],
    output_dir: str = "./outputs",
):
    os.makedirs(output_dir, exist_ok=True)

    train_x = np.arange(1, len(train_batch_losses) + 1)
    infer_x = np.arange(1, len(infer_batch_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(train_x, train_batch_losses, label="train_loss", linewidth=1.2)
    plt.plot(infer_x, infer_batch_losses, label="infer_loss", linewidth=1.2)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_x, train_batch_accs, label="train_acc", linewidth=1.2)
    plt.plot(infer_x, infer_batch_accs, label="infer_acc", linewidth=1.2)
    plt.xlabel("Batch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"), dpi=300)
    plt.close()


def run_train_and_infer(
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 256,
    num_workers: int = 0,
    data_dir: str = "./data",
):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = build_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        enable_augmentation=True,
    )

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_batch_losses_all: List[float] = []
    train_batch_accs_all: List[float] = []
    infer_batch_losses_all: List[float] = []
    infer_batch_accs_all: List[float] = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=epochs,
        )
        infer_metrics = infer_with_metrics(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            epochs=epochs,
            num_classes=10,
        )
        train_batch_losses_all.extend(train_metrics["batch_losses"])
        train_batch_accs_all.extend(train_metrics["batch_accs"])
        infer_batch_losses_all.extend(infer_metrics["batch_losses"])
        infer_batch_accs_all.extend(infer_metrics["batch_accs"])

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={train_metrics['loss']:.4f}, train_acc={train_metrics['acc']:.4f} | "
            f"infer_loss={infer_metrics['loss']:.4f}, infer_acc={infer_metrics['acc']:.4f}, "
            f"precision_macro={infer_metrics['precision_macro']:.4f}, "
            f"recall_macro={infer_metrics['recall_macro']:.4f}, "
            f"f1_macro={infer_metrics['f1_macro']:.4f}"
        )

    plot_batch_curves(
        train_batch_losses=train_batch_losses_all,
        train_batch_accs=train_batch_accs_all,
        infer_batch_losses=infer_batch_losses_all,
        infer_batch_accs=infer_batch_accs_all,
        output_dir="./outputs",
    )
    print("Saved batch curves to ./outputs/")

    return model


if __name__ == "__main__":
    run_train_and_infer(
        epochs=5, lr=1e-3, batch_size=256, num_workers=0, data_dir="./data"
    )
