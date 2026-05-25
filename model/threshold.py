"""
Dynamic Threshold Engine

Replaces the static single-value threshold with an adaptive mechanism that
computes a context-aware threshold per input based on:
  1. Confidence score    (from CNN softmax)
  2. Anomaly score       (from anomaly head)
  3. Embedding drift     (cosine distance from stored reference embeddings)
  4. Image quality proxy (Laplacian variance — blur detection)

This is one of the core novelties of the system, directly addressing the
"one-size-fits-all" limitation identified in Park et al. (2024).
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from model.config import Config

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
#  Threshold Decision Result
# ────────────────────────────────────────────────────────────────────

@dataclass
class ThresholdDecision:
    """Result produced by the DynamicThresholdEngine for a single input."""

    raw_prediction: str          # CNN raw prediction: normal / adversarial / suspicious
    final_label: str             # After threshold adjustment
    confidence_score: float
    anomaly_score: float
    embedding_drift: float
    image_quality_score: float
    computed_threshold: float
    threshold_components: dict = field(default_factory=dict)
    risk_level: str = "LOW"      # LOW / MEDIUM / HIGH

    def to_dict(self) -> dict:
        return {
            "raw_prediction": self.raw_prediction,
            "final_label": self.final_label,
            "confidence_score": self.confidence_score,
            "anomaly_score": self.anomaly_score,
            "embedding_drift": self.embedding_drift,
            "image_quality_score": self.image_quality_score,
            "computed_threshold": self.computed_threshold,
            "threshold_components": self.threshold_components,
            "risk_level": self.risk_level,
        }


# ────────────────────────────────────────────────────────────────────
#  Dynamic Threshold Engine
# ────────────────────────────────────────────────────────────────────

class DynamicThresholdEngine:
    """
    Computes an adaptive decision threshold for each input image.

    The threshold T is computed as a weighted combination:

        T = w1 * (1 - confidence)
          + w2 * anomaly_score
          + w3 * embedding_drift
          + w4 * (1 - image_quality)

    A higher T means the system is more uncertain / suspicious → the
    classification may be upgraded (normal→suspicious, suspicious→adversarial).

    Reference embeddings are stored as a gallery from clean training data
    and used to detect distributional drift in new inputs.
    """

    # Default threshold boundaries (instance-overridable via constructor)
    THRESHOLD_LOW  = 0.20    # below → keep as-is (was 0.35 — too high, T=0.26 never triggered)
    THRESHOLD_MID  = 0.40    # between → upgrade to suspicious
    THRESHOLD_HIGH = 0.65    # above → upgrade to adversarial

    def __init__(self,
                 reference_embeddings: list = None,
                 w_confidence: float = 0.35,
                 w_anomaly: float    = 0.30,
                 w_drift: float      = 0.20,
                 w_quality: float    = 0.15,
                 threshold_low: float  = None,
                 threshold_mid: float  = None,
                 threshold_high: float = None):
        """
        Args:
            reference_embeddings : list of 512-d embedding vectors from clean
                                   training images (used for drift calculation).
                                   If None, drift is set to 0.0.
            w_confidence         : weight for confidence factor (default 0.35)
            w_anomaly            : weight for anomaly score (default 0.30)
            w_drift              : weight for embedding drift (default 0.20)
            w_quality            : weight for image quality (default 0.15)
            threshold_low/mid/high: override decision boundaries per-experiment
        """
        self.W_CONFIDENCE = w_confidence
        self.W_ANOMALY    = w_anomaly
        self.W_DRIFT      = w_drift
        self.W_QUALITY    = w_quality

        self.THRESHOLD_LOW  = threshold_low  if threshold_low  is not None else DynamicThresholdEngine.THRESHOLD_LOW
        self.THRESHOLD_MID  = threshold_mid  if threshold_mid  is not None else DynamicThresholdEngine.THRESHOLD_MID
        self.THRESHOLD_HIGH = threshold_high if threshold_high is not None else DynamicThresholdEngine.THRESHOLD_HIGH

        self.reference_gallery: torch.Tensor | None = None
        if reference_embeddings:
            self.add_reference_embeddings(reference_embeddings)

    # ── Public API ──

    def compute(self, feature_dict: dict,
                image_tensor: torch.Tensor = None) -> ThresholdDecision:
        """
        Compute dynamic threshold and produce a ThresholdDecision.

        Args:
            feature_dict  : output dict from FeatureExtractor.extract()
            image_tensor  : (1, 3, 224, 224) normalized tensor (optional,
                            used for image quality estimation)

        Returns:
            ThresholdDecision
        """
        confidence   = float(feature_dict["confidence_score"])
        anomaly      = float(feature_dict["anomaly_score"])
        embedding    = feature_dict["embedding"]
        raw_pred     = feature_dict["prediction"]

        # Factor 1: confidence (inverted — low confidence = high suspicion)
        conf_factor = 1.0 - confidence

        # Factor 2: anomaly score (direct)
        anom_factor = anomaly

        # Factor 3: embedding drift
        drift = self._compute_drift(embedding)

        # Factor 4: image quality
        quality = self._image_quality(image_tensor) if image_tensor is not None else 0.5
        quality_factor = 1.0 - quality     # low quality = suspicious

        # Weighted threshold
        T = (
            self.W_CONFIDENCE * conf_factor
            + self.W_ANOMALY  * anom_factor
            + self.W_DRIFT    * drift
            + self.W_QUALITY  * quality_factor
        )
        T = float(np.clip(T, 0.0, 1.0))

        final_label = self._adjust_label(raw_pred, T)
        risk_level  = self._risk_level(T)

        return ThresholdDecision(
            raw_prediction     = raw_pred,
            final_label        = final_label,
            confidence_score   = round(confidence, 4),
            anomaly_score      = round(anomaly, 4),
            embedding_drift    = round(drift, 4),
            image_quality_score= round(quality, 4),
            computed_threshold = round(T, 4),
            threshold_components={
                "confidence_factor": round(conf_factor, 4),
                "anomaly_factor":    round(anom_factor, 4),
                "drift_factor":      round(drift, 4),
                "quality_factor":    round(quality_factor, 4),
                "weights": {
                    "w_confidence": self.W_CONFIDENCE,
                    "w_anomaly":    self.W_ANOMALY,
                    "w_drift":      self.W_DRIFT,
                    "w_quality":    self.W_QUALITY,
                },
            },
            risk_level = risk_level,
        )

    def add_reference_embeddings(self, embeddings: list[list]):
        """Add clean-image embeddings to the reference gallery."""
        new = torch.tensor(embeddings, dtype=torch.float32)
        # L2-normalize
        new = torch.nn.functional.normalize(new, p=2, dim=1)
        if self.reference_gallery is None:
            self.reference_gallery = new
        else:
            self.reference_gallery = torch.cat([self.reference_gallery, new], dim=0)
        logger.info(f"Reference gallery size: {self.reference_gallery.shape[0]}")

    # ── Private helpers ──

    def _compute_drift(self, embedding: list) -> float:
        """
        Cosine distance from the nearest reference embedding.
        Returns 0.0 if no gallery is available to prevent artificial penalty.
        """
        if self.reference_gallery is None:
            return 0.0

        emb = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        # Cosine similarity to all gallery vectors
        sims = (emb @ self.reference_gallery.T).squeeze(0)  # (N,)
        max_sim = sims.max().item()
        # Convert similarity [−1, 1] → drift [0, 1]
        drift = (1.0 - max_sim) / 2.0
        return float(np.clip(drift, 0.0, 1.0))

    @staticmethod
    def _image_quality(image_tensor: torch.Tensor) -> float:
        """
        Estimate sharpness using Laplacian variance (proxy for image quality).
        Returns value in [0, 1] where 1 = sharp / high quality.
        """
        try:
            import cv2

            # Denormalize from ImageNet normalization before computing quality.
            # Without this, values are ~[-2, 2] and Laplacian variance is ~0 for all images.
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = image_tensor.squeeze(0).cpu() * std + mean  # back to [0, 1]
            img = img.mean(dim=0).numpy()                      # grayscale (H, W)
            img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)

            lap_var = cv2.Laplacian(img_uint8, cv2.CV_64F).var()

            # Normalize: variance 0 → blurry/adversarial, ~500+ → sharp/clean
            quality = float(np.clip(lap_var / 500.0, 0.0, 1.0))
            return quality
        except Exception:
            return 0.5     # fallback: neutral quality

    def _adjust_label(self, raw_pred: str, T: float) -> str:
        """
        Upgrade the raw CNN prediction based on threshold T.

        Rules:
          T < LOW             → trust CNN prediction
          LOW ≤ T < MID       → upgrade normal → suspicious
          MID ≤ T < HIGH      → upgrade normal/suspicious → adversarial
          T ≥ HIGH            → always adversarial
        """
        if T < self.THRESHOLD_LOW:
            return raw_pred

        if T < self.THRESHOLD_MID:
            return "suspicious" if raw_pred == "normal" else raw_pred

        if T < self.THRESHOLD_HIGH:
            # suspicious and normal both upgrade to adversarial at this level
            return "adversarial" if raw_pred in ("adversarial", "suspicious") else "suspicious"

        return "adversarial"

    def _risk_level(self, T: float) -> str:
        if T < self.THRESHOLD_LOW:
            return "LOW"
        if T < self.THRESHOLD_MID:
            return "MEDIUM"
        return "HIGH"

    def format_for_agent(self, decision: ThresholdDecision) -> str:
        """Format ThresholdDecision as structured text for Advocate/Judge agents."""
        comps = decision.threshold_components
        return (
            f"[Dynamic Threshold Engine Output]\n"
            f"Raw CNN Prediction  : {decision.raw_prediction.upper()}\n"
            f"Final Label         : {decision.final_label.upper()}\n"
            f"Risk Level          : {decision.risk_level}\n"
            f"Computed Threshold  : {decision.computed_threshold:.4f}\n"
            f"Boundary (LOW/MID/HIGH): {self.THRESHOLD_LOW}/{self.THRESHOLD_MID}/{self.THRESHOLD_HIGH}\n"
            f"\nThreshold Components:\n"
            f"  Confidence Factor : {comps['confidence_factor']:.4f}  "
            f"(weight={comps['weights']['w_confidence']})\n"
            f"  Anomaly Factor    : {comps['anomaly_factor']:.4f}  "
            f"(weight={comps['weights']['w_anomaly']})\n"
            f"  Embedding Drift   : {comps['drift_factor']:.4f}  "
            f"(weight={comps['weights']['w_drift']})\n"
            f"  Quality Factor    : {comps['quality_factor']:.4f}  "
            f"(weight={comps['weights']['w_quality']})\n"
            f"\nConfidence Score    : {decision.confidence_score:.4f}\n"
            f"Anomaly Score       : {decision.anomaly_score:.4f}\n"
            f"Image Quality       : {decision.image_quality_score:.4f}\n"
        )
