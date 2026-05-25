"""
Quick inference / evaluation script.

Usage:
    # Single image
    python evaluate.py --image path/to/face.jpg --checkpoint checkpoints/best_model.pth

    # Full test-set evaluation
    python evaluate.py --csv data/labels.csv --checkpoint checkpoints/best_model.pth
"""

import argparse
import json

import torch

from model.cnn_model import FeatureExtractor
from model.config import Config
from utils.preprocessing import preprocess_image, format_features_for_agent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Adversarial Face Detector")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image")
    parser.add_argument("--csv", type=str, default=None, help="Path to labels.csv for batch eval")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def infer_single(image_path: str, checkpoint: str, device: str = None):
    extractor = FeatureExtractor(checkpoint_path=checkpoint, device=device)
    tensor = preprocess_image(image_path)
    features = extractor.extract(tensor)

    print("\n" + "=" * 55)
    print(" ADVERSARIAL FACE DETECTION — RESULT")
    print("=" * 55)
    print(format_features_for_agent(features))
    print("=" * 55)

    # Show what agents will receive
    print("\n[Agent Input Preview]")
    print(json.dumps({k: v for k, v in features.items() if k != "embedding"}, indent=2))
    return features


def batch_evaluate(csv_path: str, checkpoint: str, device: str = None):
    from model.dataset import AdversarialDataset, get_eval_transform
    from torch.utils.data import DataLoader
    from model.trainer import compute_metrics

    ds = AdversarialDataset(csv_path, transform=get_eval_transform())
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    extractor = FeatureExtractor(checkpoint_path=checkpoint, device=device)
    model = extractor.model

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(extractor.device)
            out = model(images)
            all_preds.extend(out["prediction"].cpu().tolist())
            all_labels.extend(labels.tolist())

    metrics = compute_metrics(all_preds, all_labels)
    print(f"\nBatch Evaluation on {len(ds)} samples:")
    print(f"  Accuracy : {metrics['accuracy']:.2f}%")
    print(f"  Macro F1 : {metrics['f1_macro']:.2f}%")
    for cls, f1 in metrics["f1_per_class"].items():
        print(f"  {cls:12s} : {f1:.2f}%")

    # ── FAR / FRR (biometric security metrics) ──────────────────────
    # Class indices: normal=0, adversarial=1, suspicious=2
    # FAR = adversarial images wrongly accepted as normal / total adversarial
    # FRR = normal images wrongly rejected (flagged as adv/suspicious) / total normal
    try:
        from sklearn.metrics import confusion_matrix
        import numpy as np

        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

        adv_row   = cm[1]           # row for adversarial class
        adv_total = adv_row.sum()
        FAR = adv_row[0] / adv_total if adv_total > 0 else 0.0

        norm_row   = cm[0]          # row for normal class
        norm_total = norm_row.sum()
        FRR = (norm_row[1] + norm_row[2]) / norm_total if norm_total > 0 else 0.0

        print(f"\n  ── Biometric Security Metrics ──")
        print(f"  FAR (False Acceptance Rate) : {FAR:.4f}  ({FAR*100:.2f}%)")
        print(f"  FRR (False Rejection Rate)  : {FRR:.4f}  ({FRR*100:.2f}%)")
        print(f"\n  Confusion Matrix (normal / adversarial / suspicious):")
        print(f"  {cm}")

        metrics["FAR"] = round(FAR, 4)
        metrics["FRR"] = round(FRR, 4)
        metrics["confusion_matrix"] = cm.tolist()
    except Exception as e:
        print(f"  [Warning] FAR/FRR computation failed: {e}")

    return metrics


if __name__ == "__main__":
    args = parse_args()

    if args.image:
        infer_single(args.image, args.checkpoint, args.device)
    elif args.csv:
        batch_evaluate(args.csv, args.checkpoint, args.device)
    else:
        print("Provide --image or --csv. Run with --help for details.")
