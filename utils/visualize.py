"""
Visualization utilities — plot training curves, confusion matrix, Grad-CAM heatmaps.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import cv2

from model.config import Config


# ────────────────────────────────────────────────────────────────────
#  1. Training history plots
# ────────────────────────────────────────────────────────────────────

def plot_training_history(history: dict = None,
                          history_path: str = None,
                          save_path: str = None):
    """
    Plot loss and accuracy curves from training history.

    Args:
        history     : dict returned by Trainer.fit()  (takes precedence)
        history_path: path to training_history.json   (used if history is None)
        save_path   : if given, saves figure there; otherwise shows interactively
    """
    if history is None:
        path = history_path or os.path.join(Config.LOGS_DIR, "training_history.json")
        with open(path) as f:
            history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History — Adversarial Face Detector (ResNet-18)",
                 fontsize=13, fontweight="bold")

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss",
                 color="#E74C3C", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss",
                 color="#3498DB", linewidth=2, linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc",
                 color="#2ECC71", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="Val Acc",
                 color="#9B59B6", linewidth=2, linestyle="--")
    axes[1].plot(epochs, history["val_f1"], label="Val F1 (macro)",
                 color="#F39C12", linewidth=2, linestyle=":")
    axes[1].set_title("Accuracy & F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("%")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path or os.path.join(Config.RESULTS_DIR, "training_curves.png"))


# ────────────────────────────────────────────────────────────────────
#  2. Confusion matrix
# ────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: list | np.ndarray,
                          class_names: list = None,
                          save_path: str = None):
    """
    Plot a colour-coded confusion matrix.

    Args:
        cm          : 2-D list or numpy array
        class_names : list of class labels (defaults to Config.CLASS_NAMES)
        save_path   : save file path (optional)
    """
    cm = np.array(cm)
    class_names = class_names or Config.CLASS_NAMES
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    # Overlay raw counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.72, f"({cm[i, j]})",
                    ha="center", va="center", fontsize=8, color="gray")

    ax.set_title("Confusion Matrix — Adversarial Face Detector",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    _save_or_show(fig, save_path or os.path.join(Config.RESULTS_DIR, "confusion_matrix.png"))


# ────────────────────────────────────────────────────────────────────
#  3. Grad-CAM heatmap
# ────────────────────────────────────────────────────────────────────

def compute_gradcam(model, image_tensor, target_class: int = None, device: str = "cpu"):
    """
    Compute Grad-CAM heatmap for a single image.

    Args:
        model        : AdversarialFaceDetector (eval mode)
        image_tensor : (1, 3, 224, 224) normalized tensor
        target_class : class index to visualize (None → uses predicted class)
        device       : "cuda" or "cpu"

    Returns:
        heatmap : (224, 224) numpy array, values in [0, 1]
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    image_tensor = image_tensor.to(device)

    # Hook into the last conv layer of ResNet-18 (layer4)
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations["value"] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    target_layer = model.features[-1]          # layer4
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    # Forward
    output = model(image_tensor)
    logits = output["logits"]

    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    # Backward for target class
    model.zero_grad()
    logits[0, target_class].backward()

    fh.remove()
    bh.remove()

    # Pool gradients → weights
    grads = gradients["value"]           # (1, 512, 7, 7)
    acts = activations["value"]          # (1, 512, 7, 7)
    weights = grads.mean(dim=[2, 3], keepdim=True)  # (1, 512, 1, 1)

    cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, 7, 7)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, target_class


def plot_gradcam(model, image_tensor, class_names=None,
                 target_class: int = None, device: str = "cpu",
                 save_path: str = None):
    """
    Overlay Grad-CAM heatmap on the original image.

    Args:
        model        : AdversarialFaceDetector
        image_tensor : (1, 3, 224, 224) normalized tensor
        class_names  : list of class name strings
        target_class : class index to visualize (auto-detected if None)
        device       : cuda / cpu
        save_path    : optional save path
    """
    import torch
    from utils.preprocessing import denormalize

    class_names = class_names or Config.CLASS_NAMES

    heatmap, pred_class = compute_gradcam(model, image_tensor, target_class, device)

    # Denormalize image for display
    img_np = denormalize(image_tensor.squeeze(0)).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Grad-CAM — Predicted: {class_names[pred_class].upper()}",
                 fontsize=13, fontweight="bold")

    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(img_np)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    _save_or_show(fig, save_path or os.path.join(Config.RESULTS_DIR, "gradcam.png"))
    
    if save_path:
        pure_path = save_path.replace("_gradcam.png", "_pure_overlay.png")
        if pure_path != save_path:
            heatmap_cv = np.uint8(255 * heatmap)
            color_map = cv2.applyColorMap(heatmap_cv, cv2.COLORMAP_JET)
            color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
            img_cv = np.uint8(255 * img_np)
            overlay = cv2.addWeighted(img_cv, 0.5, color_map, 0.5, 0)
            cv2.imwrite(pure_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


# ────────────────────────────────────────────────────────────────────
#  4. Clean vs adversarial comparison grid
# ────────────────────────────────────────────────────────────────────

def plot_adversarial_comparison(clean_imgs, adv_imgs,
                                n: int = 4, save_path: str = None):
    """
    Show a side-by-side grid of clean vs adversarial images.

    Args:
        clean_imgs : (N, C, H, W) tensor of clean images
        adv_imgs   : (N, C, H, W) tensor of adversarial images
        n          : number of pairs to display
        save_path  : optional save path
    """
    from utils.preprocessing import denormalize

    n = min(n, clean_imgs.size(0))
    fig, axes = plt.subplots(3, n, figsize=(n * 3, 9))
    fig.suptitle("Clean vs Adversarial Comparison", fontsize=13, fontweight="bold")

    for i in range(n):
        clean = denormalize(clean_imgs[i]).permute(1, 2, 0).numpy()
        adv = denormalize(adv_imgs[i]).permute(1, 2, 0).numpy()
        diff = np.abs(adv - clean) * 10   # amplified difference

        axes[0, i].imshow(clean)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Clean", fontsize=10)

        axes[1, i].imshow(adv)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Adversarial", fontsize=10)

        axes[2, i].imshow(np.clip(diff, 0, 1))
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_title("Perturbation (×10)", fontsize=10)

    plt.tight_layout()
    _save_or_show(fig, save_path or os.path.join(Config.RESULTS_DIR, "comparison_grid.png"))


# ────────────────────────────────────────────────────────────────────
#  Helper
# ────────────────────────────────────────────────────────────────────

def _save_or_show(fig, save_path: str):
    Path(Config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved to {save_path}")
    plt.close(fig)
