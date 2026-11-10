import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.detector import load_model
from training.dataset import DeepfakeDataset, get_validation_transforms


def evaluate_model(
    checkpoint_path: str,
    test_dir: str,
    backbone: str = "efficientnet_b0",
    batch_size: int = 16,
    output_dir: str = "evaluation_results",
    device: str = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Evaluating TruthVision on dataset: {test_dir}")

    dataset = DeepfakeDataset(test_dir, transform=get_validation_transforms(224))
    if len(dataset) == 0:
        raise ValueError(f"No test samples found in {test_dir}.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = load_model(checkpoint_path=checkpoint_path, backbone=backbone, device=device)

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            probs = model.predict_proba(images).squeeze().cpu().numpy()
            probs = np.atleast_1d(probs)

            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.50

    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "="*45)
    print("        TRUTHVISION EVALUATION REPORT        ")
    print("="*45)
    print(f"Accuracy        : {acc * 100:.2f}%")
    print(f"Precision       : {prec * 100:.2f}%")
    print(f"Recall (Sensitivity) : {rec * 100:.2f}%")
    print(f"F1-Score        : {f1 * 100:.2f}%")
    print(f"ROC-AUC         : {auc * 100:.2f}%")
    print("="*45)
    print("Confusion Matrix:")
    print(f"  True Real  (TN): {cm[0, 0]} | False Fake (FP): {cm[0, 1]}")
    print(f"  False Real (FN): {cm[1, 0]} | True Fake  (TP): {cm[1, 1]}")
    print("="*45)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Deepfake"], yticklabels=["Real", "Deepfake"])
    plt.title(f"TruthVision Confusion Matrix ({backbone.upper()})")
    plt.ylabel("Actual Ground Truth")
    plt.xlabel("Predicted Class")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#38bdf8", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#94a3b8", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("TruthVision ROC Curve")
    plt.legend(loc="lower right")
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print(f"[+] Evaluation figures saved to {output_dir}/")
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TruthVision Checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--test_dir", type=str, default="dataset/test", help="Path to test directory")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        backbone=args.backbone,
        output_dir=args.output_dir
    )
