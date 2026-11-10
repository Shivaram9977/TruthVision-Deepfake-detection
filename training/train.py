import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.detector import DeepfakeDetector
from training.dataset import DeepfakeDataset, get_training_transforms, get_validation_transforms


def train_model(
    train_dir: str,
    val_dir: str,
    backbone: str = "efficientnet_b0",
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    save_dir: str = "checkpoints",
    device: str = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[*] Initializing TruthVision Training on device: {device.upper()}")
    os.makedirs(save_dir, exist_ok=True)

    # Datasets & Loaders
    train_dataset = DeepfakeDataset(train_dir, transform=get_training_transforms(224))
    val_dataset = DeepfakeDataset(val_dir, transform=get_validation_transforms(224))

    if len(train_dataset) == 0:
        raise ValueError(f"No training samples found in {train_dir}. Structure must be train/real and train/fake.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[+] Dataset Loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # Model & Optimization
    model = DeepfakeDetector(backbone=backbone, pretrained=True, num_classes=1)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None

    best_val_loss = float("inf")
    best_val_acc = 0.0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct_train / total_train) * 100:.1f}%"})

        scheduler.step()

        train_loss = running_loss / max(1, total_train)
        train_acc = (correct_train / max(1, total_train)) * 100.0

        # Validation phase
        model.eval()
        val_loss_running = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_running += loss.item() * images.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss = val_loss_running / max(1, total_val)
        val_acc = (correct_val / max(1, total_val)) * 100.0
        epoch_time = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Checkpoint if best
        if val_loss < best_val_loss or val_acc > best_val_acc:
            best_val_loss = min(best_val_loss, val_loss)
            best_val_acc = max(best_val_acc, val_acc)
            best_path = os.path.join(save_dir, f"truthvision_{backbone}_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "backbone": backbone
            }, best_path)
            print(f"  [+] New best checkpoint saved to: {best_path}")

    print(f"[OK] Training complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TruthVision Deepfake Detection Model")
    parser.add_argument("--train_dir", type=str, default="dataset/train", help="Path to training data directory")
    parser.add_argument("--val_dir", type=str, default="dataset/validation", help="Path to validation data directory")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "resnet50", "mobilenet_v3"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    train_model(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir
    )
