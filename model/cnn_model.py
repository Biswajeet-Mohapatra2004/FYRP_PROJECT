"""
ResNet-18 Based Adversarial Face Detection Model.

This module provides:
  1. AdversarialFaceDetector  – classifies images as normal / adversarial / suspicious
  2. FeatureExtractor         – extracts 512-d embedding + anomaly metrics for agent input

The backbone is a pretrained ResNet-18 whose final FC layer is replaced with a
custom classification head.  A separate branch produces embeddings that are
forwarded to the Advocate–Judge multi-agent debate system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

from model.config import Config


# ────────────────────────────────────────────────────────────────────
#  1. Core Model – AdversarialFaceDetector
# ────────────────────────────────────────────────────────────────────

class AdversarialFaceDetector(nn.Module):
    """
    ResNet-18 backbone fine-tuned for 3-class adversarial detection.

    Architecture:
        ResNet-18 (pretrained, all layers)
            └─ avgpool → 512-d feature
                ├─ Embedding Head  → 512-d normalized embedding
                └─ Classification Head → 3 logits (normal / adversarial / suspicious)

    Forward returns:
        logits      : (B, 3)   raw class scores
        embeddings  : (B, 512) L2-normalized feature vector
        confidence  : (B,)     softmax max probability
    """

    def __init__(self, num_classes: int = Config.NUM_CLASSES,
                 pretrained: bool = Config.PRETRAINED):
        super().__init__()

        # ── Load pretrained ResNet-18 backbone ──
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            backbone = models.resnet18(weights=weights)
        else:
            backbone = models.resnet18(weights=None)

        # Everything up to (but not including) the final FC
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.avgpool = backbone.avgpool          # AdaptiveAvgPool2d → (B,512,1,1)

        feature_dim = backbone.fc.in_features    # 512 for ResNet-18

        # ── Embedding head (for agent pipeline) ──
        self.embedding_head = nn.Sequential(
            nn.Linear(feature_dim, Config.EMBEDDING_DIM),
            nn.BatchNorm1d(Config.EMBEDDING_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(Config.EMBEDDING_DIM, Config.EMBEDDING_DIM),
        )

        # ── Classification head ──
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        # ── Anomaly scoring head (scalar) ──
        self.anomaly_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),                        # outputs ∈ [0, 1]
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, 224, 224) input images

        Returns:
            dict with keys:
                logits       (B, num_classes)
                embeddings   (B, 512) – L2-normalized
                confidence   (B,)
                anomaly_score (B,)
                prediction   (B,)    – predicted class index
        """
        # Backbone feature extraction
        feat = self.features(x)                  # (B, 512, 7, 7)
        feat = self.avgpool(feat)                # (B, 512, 1, 1)
        feat = torch.flatten(feat, 1)            # (B, 512)

        # Embedding (L2-normalized for cosine similarity downstream)
        embeddings = self.embedding_head(feat)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Classification
        logits = self.classifier(feat)
        probs = F.softmax(logits, dim=1)
        confidence, prediction = probs.max(dim=1)

        # Anomaly score
        anomaly_score = self.anomaly_head(feat).squeeze(-1)

        return {
            "logits": logits,
            "embeddings": embeddings,
            "confidence": confidence,
            "anomaly_score": anomaly_score,
            "prediction": prediction,
            "probabilities": probs,
        }


# ────────────────────────────────────────────────────────────────────
#  2. Feature Extractor Wrapper (for Agent Pipeline)
# ────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    High-level wrapper that loads a trained AdversarialFaceDetector and
    produces the structured feature dict consumed by the Advocate / Judge agents.

    Usage:
        extractor = FeatureExtractor("checkpoints/best_model.pth")
        features  = extractor.extract(image_tensor)
        # features → dict ready for LangChain agent input
    """

    def __init__(self, checkpoint_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AdversarialFaceDetector().to(self.device)

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=self.device,
                               weights_only=True)
            self.model.load_state_dict(state["model_state_dict"]
                                       if "model_state_dict" in state else state)
        self.model.eval()

    @torch.no_grad()
    def extract(self, image: torch.Tensor) -> dict:
        """
        Extract features from a single image or batch.

        Args:
            image: (3, 224, 224) or (B, 3, 224, 224) tensor, already normalized.

        Returns:
            dict with human-readable keys for the agent pipeline:
                prediction        : str   ("normal" / "adversarial" / "suspicious")
                confidence_score  : float
                anomaly_score     : float
                embedding         : list[float]  (512-d)
                class_probabilities : dict  {class_name: prob}
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        output = self.model(image)

        pred_idx = output["prediction"].item()
        pred_label = Config.CLASS_NAMES[pred_idx]
        probs = output["probabilities"].squeeze(0).cpu().tolist()

        return {
            "prediction": pred_label,
            "confidence_score": round(output["confidence"].item(), 4),
            "anomaly_score": round(output["anomaly_score"].item(), 4),
            "embedding": output["embeddings"].squeeze(0).cpu().tolist(),
            "class_probabilities": {
                name: round(p, 4)
                for name, p in zip(Config.CLASS_NAMES, probs)
            },
        }

    @torch.no_grad()
    def extract_batch(self, images: torch.Tensor) -> list[dict]:
        """Extract features for an entire batch."""
        images = images.to(self.device)
        output = self.model(images)
        results = []
        for i in range(images.size(0)):
            pred_idx = output["prediction"][i].item()
            probs = output["probabilities"][i].cpu().tolist()
            results.append({
                "prediction": Config.CLASS_NAMES[pred_idx],
                "confidence_score": round(output["confidence"][i].item(), 4),
                "anomaly_score": round(output["anomaly_score"][i].item(), 4),
                "embedding": output["embeddings"][i].cpu().tolist(),
                "class_probabilities": {
                    name: round(p, 4)
                    for name, p in zip(Config.CLASS_NAMES, probs)
                },
            })
        return results


# ────────────────────────────────────────────────────────────────────
#  3. Utility – model summary
# ────────────────────────────────────────────────────────────────────

def model_summary(model: nn.Module = None):
    """Print a quick parameter count summary."""
    if model is None:
        model = AdversarialFaceDetector()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {Config.MODEL_NAME}-based AdversarialFaceDetector")
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}")
    print(f"  Classes          : {Config.CLASS_NAMES}")
    print(f"  Embedding dim    : {Config.EMBEDDING_DIM}")
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    model_summary()
    # Quick forward pass test
    dummy = torch.randn(2, 3, 224, 224)
    model = AdversarialFaceDetector()
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print(f"\nForward pass test:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:20s} → shape {tuple(v.shape)}")
