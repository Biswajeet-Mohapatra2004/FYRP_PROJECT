"""
Training Engine for the Adversarial Face Detector.

Handles:
  - Full training loop with validation
  - Early stopping
  - LR scheduling
  - Checkpoint saving (best + last)
  - Per-epoch metrics logging (accuracy, loss, per-class F1)
  - Final evaluation on test set
"""

import os
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)
import numpy as np

from model.config import Config
from model.cnn_model import AdversarialFaceDetector

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
#  Metrics helper
# ────────────────────────────────────────────────────────────────────

def compute_metrics(all_preds: list, all_labels: list) -> dict:
    """Compute accuracy, per-class F1, macro-F1."""
    acc = accuracy_score(all_labels, all_preds) * 100
    f1_macro = f1_score(all_labels, all_preds, average="macro",
                        zero_division=0) * 100
    f1_per_class = f1_score(all_labels, all_preds, average=None,
                            zero_division=0)
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(len(Config.CLASS_NAMES))),
        target_names=Config.CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": round(acc, 3),
        "f1_macro": round(f1_macro, 3),
        "f1_per_class": {
            name: round(float(f1_per_class[i]) * 100, 3)
            for i, name in enumerate(Config.CLASS_NAMES)
            if i < len(f1_per_class)
        },
        "report": report,
    }


# ────────────────────────────────────────────────────────────────────
#  Trainer
# ────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Full training loop for AdversarialFaceDetector.

    Usage:
        trainer = Trainer()
        history = trainer.fit(train_loader, val_loader)
        results = trainer.evaluate(test_loader)
        trainer.save("checkpoints/best_model.pth")
    """

    def __init__(self,
                 model: AdversarialFaceDetector = None,
                 device: str = None,
                 learning_rate: float = Config.LEARNING_RATE,
                 weight_decay: float = Config.WEIGHT_DECAY,
                 class_weights: torch.Tensor = None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {self.device}")

        self.model = (model or AdversarialFaceDetector()).to(self.device)

        # Classification Loss — optionally weighted for imbalanced datasets
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device) if class_weights is not None else None
        )
        
        # Anomaly Loss
        self.anomaly_criterion = nn.BCELoss()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = StepLR(
            self.optimizer,
            step_size=Config.SCHEDULER_STEP,
            gamma=Config.SCHEDULER_GAMMA,
        )

        self.best_val_acc: float = 0.0
        self.patience_counter: int = 0
        self.history: dict = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_f1": [],
        }

        # Save dir
        Path(Config.MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
        Path(Config.LOGS_DIR).mkdir(parents=True, exist_ok=True)

    # ── Training loop ──

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = Config.NUM_EPOCHS) -> dict:
        """
        Run the full training loop.

        Returns:
            history dict: {train_loss, train_acc, val_loss, val_acc, val_f1}
        """
        print(f"\n{'='*60}")
        print(f" Training AdversarialFaceDetector (ResNet-18)")
        print(f" Epochs: {epochs} | Batch: {Config.BATCH_SIZE} | LR: {self.optimizer.param_groups[0]['lr']}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # ── Train one epoch ──
            train_loss, train_acc = self._train_epoch(train_loader)

            # ── Validate ──
            val_loss, val_acc, val_metrics = self._val_epoch(val_loader)
            val_f1 = val_metrics["f1_macro"]

            self.scheduler.step()
            elapsed = time.time() - t0

            # ── Log ──
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)

            print(f"Epoch [{epoch:3d}/{epochs}]  "
                  f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%  |  "
                  f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%  "
                  f"F1: {val_f1:.2f}%  "
                  f"[{elapsed:.1f}s]")

            # ── Checkpoint best model ──
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_acc, "best_model.pth")
                print(f"  [Best Model] saved (val_acc={val_acc:.2f}%)")
            else:
                self.patience_counter += 1

            # ── Early stopping ──
            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\n[Early Stopping] at epoch {epoch} "
                      f"(no improvement for {Config.EARLY_STOPPING_PATIENCE} epochs)")
                break

        # Save last checkpoint
        self._save_checkpoint(epoch, val_acc, "last_model.pth")
        self._save_history()
        print(f"\n Training complete. Best val accuracy: {self.best_val_acc:.2f}%")
        return self.history

    # ── Single train epoch ──

    def _train_epoch(self, loader: DataLoader):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(images)
            
            # 1. Classification Loss
            cls_loss = self.criterion(output["logits"], labels)
            
            # 2. Anomaly Loss (0 if clean, 1 if adversarial/suspicious)
            anomaly_targets = (labels != 0).float()
            anom_loss = self.anomaly_criterion(output["anomaly_score"], anomaly_targets)
            
            # Combine losses
            loss = cls_loss + anom_loss
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (output["prediction"] == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total * 100

    # ── Single validation epoch ──

    def _val_epoch(self, loader: DataLoader):
        self.model.eval()
        total_loss, total = 0.0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                output = self.model(images)
                
                # 1. Classification Loss
                cls_loss = self.criterion(output["logits"], labels)
                
                # 2. Anomaly Loss
                anomaly_targets = (labels != 0).float()
                anom_loss = self.anomaly_criterion(output["anomaly_score"], anomaly_targets)
                
                # Combine losses
                loss = cls_loss + anom_loss

                total_loss += loss.item() * images.size(0)
                all_preds.extend(output["prediction"].cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                total += labels.size(0)

        metrics = compute_metrics(all_preds, all_labels)
        return total_loss / total, metrics["accuracy"], metrics

    # ── Test evaluation ──

    def evaluate(self, test_loader: DataLoader) -> dict:
        """Full evaluation on the test set. Loads best checkpoint automatically."""
        best_path = os.path.join(Config.MODEL_SAVE_DIR, "best_model.pth")
        if os.path.exists(best_path):
            state = torch.load(best_path, map_location=self.device,
                               weights_only=True)
            self.model.load_state_dict(state["model_state_dict"])
            print(f"Loaded best model from {best_path}")

        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)
                all_preds.extend(output["prediction"].cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        metrics = compute_metrics(all_preds, all_labels)
        cm = confusion_matrix(all_labels, all_preds)

        print(f"\n{'='*60}")
        print(f" TEST SET RESULTS")
        print(f"{'='*60}")
        print(f" Accuracy : {metrics['accuracy']:.2f}%")
        print(f" Macro F1 : {metrics['f1_macro']:.2f}%")
        print(f"\n Per-class F1:")
        for cls, f1 in metrics["f1_per_class"].items():
            print(f"   {cls:12s} : {f1:.2f}%")
        print(f"\n Confusion Matrix:\n{cm}")
        print(f"{'='*60}\n")

        # Save results
        results = {**metrics, "confusion_matrix": cm.tolist()}
        results_path = os.path.join(Config.RESULTS_DIR, "test_results.json")
        Path(Config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")
        return results

    # ── Checkpointing ──

    def _save_checkpoint(self, epoch: int, val_acc: float, filename: str):
        path = os.path.join(Config.MODEL_SAVE_DIR, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_accuracy": val_acc,
            "config": {
                "num_classes": Config.NUM_CLASSES,
                "class_names": Config.CLASS_NAMES,
                "embedding_dim": Config.EMBEDDING_DIM,
                "model_name": Config.MODEL_NAME,
            }
        }, path)

    def _save_history(self):
        path = os.path.join(Config.LOGS_DIR, "training_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
