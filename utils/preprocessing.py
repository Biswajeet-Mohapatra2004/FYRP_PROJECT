"""
Preprocessing utilities for facial images.

Provides:
  - preprocess_image()    : single image → normalized tensor (for inference)
  - denormalize()         : undo ImageNet normalization (for visualization)
  - compute_embedding_distance() : cosine distance between two embeddings
"""

import io
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model.config import Config

try:
    from facenet_pytorch import MTCNN
    _MTCNN = MTCNN(keep_all=False, device='cpu')
except ImportError:
    _MTCNN = None

# ────────────────────────────────────────────────────────────────────
#  Transforms
# ────────────────────────────────────────────────────────────────────

_INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=Config.NORMALIZE_MEAN,
                         std=Config.NORMALIZE_STD),
])


def preprocess_image(source) -> torch.Tensor:
    """
    Load and preprocess a single image for model inference.

    Args:
        source: file path (str/Path), PIL Image, raw bytes, or numpy array.

    Returns:
        Tensor of shape (1, 3, 224, 224), normalized.
    """
    if isinstance(source, (str, Path)):
        img = Image.open(source).convert("RGB")
    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source)).convert("RGB")
    elif isinstance(source, np.ndarray):
        img = Image.fromarray(source).convert("RGB")
    elif isinstance(source, Image.Image):
        img = source.convert("RGB")
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    if _MTCNN is not None:
        try:
            boxes, probs = _MTCNN.detect(img)
            if boxes is not None and len(boxes) > 0:
                box = boxes[0]
                margin = 40
                x1 = max(0, box[0] - margin)
                y1 = max(0, box[1] - margin)
                x2 = min(img.width, box[2] + margin)
                y2 = min(img.height, box[3] + margin)
                img = img.crop((x1, y1, x2, y2))
        except Exception:
            pass # Fallback to original image if MTCNN fails

    return _INFERENCE_TRANSFORM(img).unsqueeze(0)   # (1, 3, 224, 224)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Undo ImageNet normalization so the tensor can be visualized.

    Args:
        tensor: (C, H, W) or (B, C, H, W)

    Returns:
        Tensor with values clamped to [0, 1]
    """
    mean = torch.tensor(Config.NORMALIZE_MEAN, device=tensor.device)
    std = torch.tensor(Config.NORMALIZE_STD, device=tensor.device)

    if tensor.dim() == 4:           # (B, C, H, W)
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:                           # (C, H, W)
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)

    return torch.clamp(tensor * std + mean, 0.0, 1.0)


# ────────────────────────────────────────────────────────────────────
#  Embedding utilities
# ────────────────────────────────────────────────────────────────────

def compute_embedding_distance(emb_a: list | torch.Tensor,
                               emb_b: list | torch.Tensor,
                               metric: str = "cosine") -> float:
    """
    Compute distance between two 512-d embedding vectors.

    Args:
        emb_a, emb_b : list[float] or 1-D Tensor
        metric       : "cosine" (default) or "euclidean"

    Returns:
        float distance score
    """
    a = torch.tensor(emb_a, dtype=torch.float32) if isinstance(emb_a, list) else emb_a.float()
    b = torch.tensor(emb_b, dtype=torch.float32) if isinstance(emb_b, list) else emb_b.float()

    if metric == "cosine":
        # cosine distance ∈ [0, 2]; 0 = identical, 2 = opposite
        sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
        return round((1.0 - sim.item()), 6)
    elif metric == "euclidean":
        return round(torch.dist(a, b).item(), 6)
    else:
        raise ValueError(f"Unknown metric: {metric!r}. Use 'cosine' or 'euclidean'.")


def format_features_for_agent(feature_dict: dict) -> str:
    """
    Convert the FeatureExtractor output dict into a structured text block
    that can be injected into LangChain agent prompts.

    Args:
        feature_dict: output of FeatureExtractor.extract()

    Returns:
        Formatted string for LLM prompt context.
    """
    probs = feature_dict.get("class_probabilities", {})
    prob_lines = "\n".join(
        f"  - {cls}: {prob * 100:.2f}%"
        for cls, prob in probs.items()
    )
    return (
        f"[CNN Model Output]\n"
        f"Prediction      : {feature_dict.get('prediction', 'N/A').upper()}\n"
        f"Confidence Score: {feature_dict.get('confidence_score', 0):.4f}  "
        f"({feature_dict.get('confidence_score', 0) * 100:.2f}%)\n"
        f"Anomaly Score   : {feature_dict.get('anomaly_score', 0):.4f}  "
        f"({'HIGH' if feature_dict.get('anomaly_score', 0) > 0.5 else 'LOW'} risk)\n"
        f"Class Probabilities:\n{prob_lines}\n"
        f"Embedding Norm  : "
        f"{torch.tensor(feature_dict.get('embedding', [0])).norm().item():.4f}\n"
    )
