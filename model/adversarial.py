"""
Adversarial Attack Generation Module.

Implements:
  - FGSM  (Fast Gradient Sign Method)      – single-step, fast
  - PGD   (Projected Gradient Descent)     – iterative, stronger
  - AdversarialDatasetGenerator            – full pipeline to generate
                                              and save clean/adversarial splits

These attacks operate on a *trained* CNN model and a *clean* dataset to
produce adversarial examples that are then saved to disk and indexed in a CSV
for training the final 3-class detector.
"""

import os
import csv
import copy
import logging
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from model.config import Config

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
#  1. FGSM Attack
# ────────────────────────────────────────────────────────────────────

class FGSM:
    """
    Fast Gradient Sign Method (Goodfellow et al., 2015).

    x_adv = x + ε · sign(∇_x L(f(x), y))

    Single-step, cheap, less powerful than PGD but fast for dataset generation.
    """

    def __init__(self, model: nn.Module, epsilon: float = Config.FGSM_EPSILON):
        self.model = model
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()

    def perturb(self, images: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute adversarial examples for a batch.

        Args:
            images : (B, C, H, W) normalized input images (ImageNet mean/std)
            labels : (B,)          true class indices

        Returns:
            adv_images : (B, C, H, W) normalized adversarial images
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        mean = torch.tensor(Config.NORMALIZE_MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(device)
        std  = torch.tensor(Config.NORMALIZE_STD,  dtype=torch.float32).view(1, 3, 1, 1).to(device)

        # Compute gradient in normalized space (model expects normalized input)
        x = images.clone().detach().requires_grad_(True).to(device)
        labels = labels.to(device)

        output = self.model(x)
        logits = output["logits"] if isinstance(output, dict) else output
        loss = self.criterion(logits, labels)

        self.model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad_sign = x.grad.sign()
            # Apply epsilon in pixel space [0, 1]: denorm → perturb → clamp → renorm
            img_pixel = images.to(device) * std + mean
            adv_pixel = torch.clamp(img_pixel + self.epsilon * grad_sign, 0.0, 1.0)
            adv_images = (adv_pixel - mean) / std

        return adv_images.detach()


# ────────────────────────────────────────────────────────────────────
#  2. PGD Attack
# ────────────────────────────────────────────────────────────────────

class PGD:
    """
    Projected Gradient Descent (Madry et al., 2018).

    Stronger than FGSM: applies FGSM iteratively and projects
    the perturbation back into an ε-ball after each step.

    x_0   = x + Uniform(-ε, ε)   (random start)
    x_{t+1} = Π_{ε}(x_t + α · sign(∇_{x_t} L))
    """

    def __init__(self, model: nn.Module,
                 epsilon: float = Config.PGD_EPSILON,
                 alpha: float = Config.PGD_ALPHA,
                 steps: int = Config.PGD_STEPS):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.criterion = nn.CrossEntropyLoss()

    def perturb(self, images: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute PGD adversarial examples.

        Args:
            images : (B, C, H, W) normalized input images (ImageNet mean/std)
            labels : (B,)          true class indices

        Returns:
            adv_images : (B, C, H, W) normalized adversarial images
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        images = images.to(device)
        labels = labels.to(device)

        mean = torch.tensor(Config.NORMALIZE_MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(device)
        std  = torch.tensor(Config.NORMALIZE_STD,  dtype=torch.float32).view(1, 3, 1, 1).to(device)

        # Work in pixel space [0, 1] for correct epsilon semantics
        img_pixel = images * std + mean

        # Random start within ε-ball in pixel space
        adv_pixel = img_pixel.clone().detach()
        adv_pixel += torch.empty_like(adv_pixel).uniform_(-self.epsilon, self.epsilon)
        adv_pixel = torch.clamp(adv_pixel, 0.0, 1.0)

        for _ in range(self.steps):
            # Re-normalize for model input
            adv_norm = ((adv_pixel - mean) / std).requires_grad_(True)

            output = self.model(adv_norm)
            logits = output["logits"] if isinstance(output, dict) else output
            loss = self.criterion(logits, labels)

            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                grad_sign = adv_norm.grad.sign()
                adv_pixel = adv_pixel + self.alpha * grad_sign

                # Project back into ε-ball around original in pixel space
                delta = torch.clamp(adv_pixel - img_pixel, -self.epsilon, self.epsilon)
                adv_pixel = torch.clamp(img_pixel + delta, 0.0, 1.0).detach()

        return ((adv_pixel - mean) / std).detach()


# ────────────────────────────────────────────────────────────────────
#  3. Adversarial Dataset Generator
# ────────────────────────────────────────────────────────────────────

class AdversarialDatasetGenerator:
    """
    Runs FGSM and PGD over a clean DataLoader and saves:
      - clean images       →  data/clean/
      - adversarial images →  data/adversarial/fgsm/
                               data/adversarial/pgd/
      - labels.csv         →  data/labels.csv

    CSV schema:
        image_path, label, image_type, attack_method
        data/clean/img_0000.png, 0, clean, none
        data/adversarial/fgsm/img_0000.png, 1, adversarial, fgsm
    """

    def __init__(self, model: nn.Module,
                 clean_loader: DataLoader,
                 output_dir: str = Config.DATA_DIR,
                 attack_types: list = None,
                 device: str = None):
        self.model = model
        self.loader = clean_loader
        self.output_dir = Path(output_dir)
        self.attack_types = attack_types or ["fgsm", "pgd"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Build attack objects
        self.attacks: dict[str, FGSM | PGD] = {}
        if "fgsm" in self.attack_types:
            self.attacks["fgsm"] = FGSM(self.model)
        if "pgd" in self.attack_types:
            self.attacks["pgd"] = PGD(self.model)

        # Output paths
        self.clean_dir = self.output_dir / "clean"
        self.adv_dirs = {
            name: self.output_dir / "adversarial" / name
            for name in self.attacks
        }
        self.csv_path = self.output_dir / "labels.csv"

    def _mkdir(self):
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        for d in self.adv_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def generate(self, denormalize: bool = True) -> str:
        """
        Run generation over the entire DataLoader.

        Args:
            denormalize: if True, undo ImageNet normalization before saving
                         so saved images look correct visually.

        Returns:
            path to the generated CSV file
        """
        self._mkdir()
        self.model.eval()

        mean = torch.tensor(Config.NORMALIZE_MEAN).view(3, 1, 1).to(self.device)
        std = torch.tensor(Config.NORMALIZE_STD).view(3, 1, 1).to(self.device)

        rows: list[dict] = []
        img_idx = 0

        logger.info(f"Generating adversarial dataset -> {self.output_dir}")

        for images, labels in tqdm(self.loader, desc="Generating"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)

            for b in range(batch_size):
                fname = f"img_{img_idx:06d}.png"

                # ── Save clean image ──
                clean_img = images[b]
                clean_save = self._maybe_denorm(clean_img, mean, std, denormalize)
                save_image(clean_save, self.clean_dir / fname)
                rows.append({
                    "image_path": str(self.clean_dir / fname),
                    "label": 0,
                    "image_type": "clean",
                    "attack_method": "none",
                })

                img_idx += 1

            # ── Save adversarial images ──
            for attack_name, attack in self.attacks.items():
                adv_images = attack.perturb(images, labels)
                for b in range(batch_size):
                    fname = f"img_{img_idx - batch_size + b:06d}.png"
                    adv_img = self._maybe_denorm(adv_images[b], mean, std, denormalize)
                    save_path = self.adv_dirs[attack_name] / fname
                    save_image(adv_img, save_path)
                    rows.append({
                        "image_path": str(save_path),
                        "label": 1,
                        "image_type": "adversarial",
                        "attack_method": attack_name,
                    })

        # ── Write CSV ──
        fieldnames = ["image_path", "label", "image_type", "attack_method"]
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        total = len(rows)
        clean_count = sum(1 for r in rows if r["image_type"] == "clean")
        adv_count = total - clean_count
        logger.info(f"Done. {clean_count} clean + {adv_count} adversarial -> {self.csv_path}")
        print(f"\n[Done] Dataset generated:")
        print(f"   Clean images      : {clean_count}")
        print(f"   Adversarial images: {adv_count}")
        print(f"   CSV saved to      : {self.csv_path}")
        return str(self.csv_path)

    @staticmethod
    def _maybe_denorm(img: torch.Tensor,
                      mean: torch.Tensor,
                      std: torch.Tensor,
                      denormalize: bool) -> torch.Tensor:
        if denormalize:
            return torch.clamp(img * std + mean, 0.0, 1.0)
        return img


# ────────────────────────────────────────────────────────────────────
#  4. Quick attack evaluation helper
# ────────────────────────────────────────────────────────────────────

def evaluate_attack(model: nn.Module,
                    loader: DataLoader,
                    attack: Literal["fgsm", "pgd"] = "fgsm",
                    device: str = None) -> dict:
    """
    Report accuracy drop from a single-attack evaluation pass.

    Returns:
        dict with clean_accuracy, adv_accuracy, attack_success_rate
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    attacker = FGSM(model) if attack == "fgsm" else PGD(model)
    clean_correct = adv_correct = total = 0

    for images, labels in tqdm(loader, desc=f"Evaluating {attack.upper()}"):
        images, labels = images.to(device), labels.to(device)
        adv_images = attacker.perturb(images, labels)

        with torch.no_grad():
            clean_out = model(images)
            adv_out = model(adv_images)

            c_preds = (clean_out["prediction"] if isinstance(clean_out, dict)
                       else clean_out.argmax(1))
            a_preds = (adv_out["prediction"] if isinstance(adv_out, dict)
                       else adv_out.argmax(1))

        clean_correct += (c_preds == labels).sum().item()
        adv_correct += (a_preds == labels).sum().item()
        total += labels.size(0)

    clean_acc = clean_correct / total * 100
    adv_acc = adv_correct / total * 100
    return {
        "clean_accuracy": round(clean_acc, 2),
        "adv_accuracy": round(adv_acc, 2),
        "attack_success_rate": round(clean_acc - adv_acc, 2),
    }
