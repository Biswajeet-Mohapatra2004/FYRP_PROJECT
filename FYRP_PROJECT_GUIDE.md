# 🛡️ FYRP: Complete Project Guide
**Explainable Multi-Agent Adversarial Threat Detection for Facial Biometric Systems**

---

## 📁 Project Structure

```
FYRP_codebase/
├── train.py                  # Main training script (3 modes)
├── pipeline.py               # End-to-end inference orchestrator
├── extract_casia.py          # Dataset extraction from RecordIO format
├── setup_lfw.py              # Downloads & extracts LFW "in-the-wild" images
├── download_casia.py         # Dataset downloader (Kaggle)
├── generate_test_attacks.py  # CLI script to quickly test FGSM/PGD on an image
├── fyrp_comprehensive_walkthrough.ipynb  # Interactive project notebook
├── .env                      # API keys & LLM config (gitignored)
├── .env.example              # Template for .env
│
├── model/
│   ├── config.py             # ← ALL hyperparameters live here
│   ├── cnn_model.py          # ResNet-18 CNN architecture
│   ├── dataset.py            # DataLoaders for clean & adversarial data
│   ├── adversarial.py        # FGSM & PGD attack implementations
│   ├── threshold.py          # Dynamic Threshold Engine
│   └── trainer.py            # Training loop, early stopping, metrics
│
├── agents/
│   ├── llm_factory.py        # ← Single point to switch LLM providers
│   ├── advocate.py           # Proponent & Opponent agents
│   └── judge.py              # Judge agent (LLM-as-a-Judge)
│
├── utils/
│   ├── preprocessing.py      # MTCNN alignment & tensor normalization
│   └── visualize.py          # Grad-CAM, training curves, confusion matrix
│
├── checkpoints/
│   └── best_model.pth        # Saved model weights (produced by training)
│
├── data/
│   ├── clean/                # Clean face images (identity subfolders)
│   └── adversarial/          # Generated FGSM/PGD images
│
└── results/
    ├── *_report.json         # Per-image decision report
    └── *_gradcam.png         # Grad-CAM heatmap
```

---

## 🚀 Step-by-Step Execution Guide

### Step 0 — Prerequisites

- **OS:** Windows 10/11
- **GPU:** NVIDIA with CUDA 12.1+ (tested on RTX 3060)
- **Python:** 3.10 or 3.11
- **Kaggle Account:** for downloading CASIA-WebFace

---

### Step 1 — Create & Activate Virtual Environment

We recommend using the provided script on Windows to automatically handle a known dependency conflict between `torchattacks` and `langchain`:

```powershell
.\setup_venv.bat
```

**Manual Installation (if not using the script):**
```powershell
# Create the venv
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Install dependencies (ignoring the torchattacks conflict)
pip install -r requirements.txt
pip install "torchattacks>=3.5.1" --no-deps  # Important: bypasses 'requests' version conflict

# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Verify GPU is detected:**
```powershell
venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True | NVIDIA GeForce RTX 3060
```

---

### Step 2 — Configure API Keys

Copy the template and fill in your details:
```powershell
Copy-Item .env.example .env
```

Edit `.env`:
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
LLM_MODEL=llama-3.3-70b-versatile
```

> **To switch to OpenAI:** Change `LLM_PROVIDER=openai` and add `OPENAI_API_KEY=sk-...` — no code changes needed.

---

### Step 3 — Download & Extract Dataset

**Option A: Download fresh from Kaggle**
```powershell
# 1. Get your Kaggle API key from https://kaggle.com → Account → API → Create New Token
# 2. Place kaggle.json at: C:\Users\<YourUsername>\.kaggle\kaggle.json

# Download (~2.7 GB)
venv\Scripts\python.exe download_casia.py

# Extract to data/clean/ (~490K images, 15-20 minutes)
venv\Scripts\python.exe extract_casia.py
```

**Option B: Use 10K subset (Recommended — faster)**

The default `extract_casia.py` is already configured for the 10K/2K subset.
Just run:
```powershell
venv\Scripts\python.exe extract_casia.py
```

**Verify extraction:**
```powershell
venv\Scripts\python.exe -c "import os; ids=os.listdir('data/clean'); print(len(ids), 'identities extracted')"
```

**Step 3.5 — LFW "In-The-Wild" Data Equalization**
To ensure the Anomaly Head doesn't falsely flag perfectly normal selfies, extract 10,000 clean LFW images to match CASIA:
```powershell
venv\Scripts\python.exe -c "import setup_lfw; setup_lfw.setup_lfw_subset(10000)"
```

---

### Step 4 — Phase 1 Training (Clean Data)

This trains the ResNet-18 backbone to learn normal facial features:

```powershell
venv\Scripts\python.exe train.py --mode clean --data_dir data/clean --epochs 30 --batch_size 32
```

- **Output:** `checkpoints/best_model.pth`
- **Time:** ~3-5 hours on RTX 3060 for 30 epochs
- **Watch for:** Validation accuracy improving, early stopping if patience (7 epochs) exceeded

> ⚠️ **Windows DataLoader fix:** If it hangs after "Starting training...", `NUM_WORKERS` in `model/config.py` is already set to `0`. This is safe.

---

### Step 5 — Generate Adversarial Examples (FGSM + PGD)

Using the Phase 1 model, generate 20,000 adversarial attack images:

```powershell
venv\Scripts\python.exe train.py --mode generate --data_dir data/clean --checkpoint checkpoints/best_model.pth
```

- **Output:**
  - `data/adversarial/fgsm/` — FGSM attacked images
  - `data/adversarial/pgd/` — PGD attacked images
  - `data/labels.csv` — Full manifest (image path, label, attack type)

---

### Step 6 — Phase 2 Training (Adversarial Fine-Tuning)

Fine-tune as a 3-class detector (normal / adversarial / suspicious):

```powershell
venv\Scripts\python.exe train.py --mode adversarial --csv data/labels.csv --checkpoint checkpoints/best_model.pth --epochs 20 --batch_size 32
```

- **Output:** `checkpoints/best_model.pth` (updated)
- **Our result:** 100% test accuracy on the 10K subset

---

### Step 7 — Run Full Pipeline on a Test Image

```powershell
venv\Scripts\python.exe pipeline.py --image images.jpg --checkpoint checkpoints/best_model.pth
```

**Output:**
- `results/images_report.json` — Full JSON decision report with agent reasoning
- `results/images_gradcam.png` — Grad-CAM heatmap showing WHERE the attack was detected
- `results/images_dashboard.png` — The Unified Dashboard rendering everything together

---

### Step 8 — Test Attacks Independently

To generate adversarial attacks on your own images dynamically outside of training:
```powershell
python generate_test_attacks.py my_face.jpg
```
This produces `my_face_fgsm.jpg` and `my_face_pgd.jpg`.

---

### Step 8 — (Optional) Evaluate on a Batch

```powershell
venv\Scripts\python.exe evaluate.py --image images.jpg --checkpoint checkpoints/best_model.pth
```

---

## 🎛️ Customization Reference

All major customization points are listed below. Most are controlled from a **single file**.

---

### 🔧 1. Hyperparameters — `model/config.py`

This is the **single source of truth** for all settings. Change one value here and it propagates everywhere.

```python
class Config:
    # ── Dataset ──────────────────────────────────────────────────────
    IMAGE_SIZE = 224          # Change to 112, 96, etc. for smaller models
    BATCH_SIZE = 32           # Increase to 64 if you have more VRAM
    VALIDATION_SPLIT = 0.2    # Fraction of data held out for validation

    # ── Training ─────────────────────────────────────────────────────
    NUM_EPOCHS = 30           # Increase for full dataset training
    LEARNING_RATE = 1e-4      # Tune for your dataset
    WEIGHT_DECAY = 1e-4       # L2 regularization
    EARLY_STOPPING_PATIENCE = 7  # Stop if val_acc doesn't improve for N epochs
    SCHEDULER_STEP = 10       # LR drops every N epochs
    SCHEDULER_GAMMA = 0.1     # LR multiplied by this on each step

    # ── Adversarial Attacks ──────────────────────────────────────────
    FGSM_EPSILON = 0.03       # Perturbation budget (higher = stronger attack)
    PGD_EPSILON = 0.03        # PGD perturbation budget
    PGD_ALPHA = 0.007         # PGD step size
    PGD_STEPS = 10            # PGD iterations (more = stronger)

    # ── Dynamic Threshold Weights ─────────────────────────────────── 
    # (in model/threshold.py — tune these if your sensor has known biases)
    W_CONFIDENCE = 0.35
    W_ANOMALY    = 0.30
    W_DRIFT      = 0.20
    W_QUALITY    = 0.15
```

---

### 🏗️ 2. Swap the CNN Backbone — `model/cnn_model.py`

Currently: **ResNet-18** (pretrained on ImageNet).

To use a different backbone, edit `cnn_model.py`:

```python
# Current (ResNet-18)
backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# ─── To switch to ResNet-50 (larger, more accurate, slower) ───
from torchvision.models import ResNet50_Weights
backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
feature_dim = 2048  # ResNet-50 has 2048-d features (not 512)

# ─── To switch to EfficientNet-B0 (mobile-friendly) ──────────
from torchvision.models import EfficientNet_B0_Weights
backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
feature_dim = 1280  # EfficientNet-B0 has 1280-d features
```

> After changing backbone, update `feature_dim` and retrain from scratch.

---

### 📦 3. Swap the Dataset

To use a **different face dataset** (e.g., LFW, VGGFace2, MS-Celeb-1M):

1. Organize images in the same folder structure: `data/clean/<identity_id>/<image>.jpg`
2. Run `train.py --mode clean --data_dir data/clean` — the `FaceDataset` class in `model/dataset.py` reads from any identity-subfolder layout.

To change the **subset size**, edit `extract_casia.py`:
```python
MAX_IDENTITIES = 2000   # Increase for full dataset
MAX_IMGS_PER_ID = 5     # Images per identity
```

---

### ⚔️ 4. Add New Attack Types — `model/adversarial.py`

The FGSM and PGD classes follow a simple interface: implement `perturb(images, labels)` and return perturbed images.

To add **CW (Carlini-Wagner)** or **AutoAttack**, create a new class following this pattern:

```python
class CarliniWagner:
    def __init__(self, model, c=1e-4, kappa=0, steps=1000, lr=0.01):
        self.model = model
        # ... your CW params

    def perturb(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # ... CW optimization loop
        return adv_images
```

Then register it in `AdversarialDatasetGenerator.__init__`:
```python
if "cw" in self.attack_types:
    self.attacks["cw"] = CarliniWagner(self.model)
```

And add it to `train.py` generate mode:
```powershell
python train.py --mode generate --attack_types fgsm pgd cw
```

---

### 🤖 5. Switch LLM Provider — `.env` only

No code changes needed. Just edit `.env`:

```env
# Use Groq (free, fast)
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
LLM_MODEL=llama-3.3-70b-versatile

# ─── Switch to OpenAI GPT-4o ───
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o
```

To add a **completely new provider** (e.g., Anthropic Claude), only edit `agents/llm_factory.py`:

```python
elif provider == "anthropic":
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=model_name or "claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=temperature,
    )
```

---

### 📊 6. Change the Dynamic Threshold Formula — `model/threshold.py`

The threshold formula is:
```
T = 0.35*(1-confidence) + 0.30*anomaly + 0.20*drift + 0.15*(1-quality)
```

To adjust weights (must sum to 1.0):
```python
class DynamicThresholdEngine:
    W_CONFIDENCE = 0.35   # Increase if CNN confidence is very reliable
    W_ANOMALY    = 0.30   # Increase for anomaly-heavy threat scenarios
    W_DRIFT      = 0.20   # Increase if gallery drift is a key signal
    W_QUALITY    = 0.15   # Increase for low-quality camera inputs

    THRESHOLD_LOW  = 0.25  # Below this → trust CNN as-is (LOW risk)
    THRESHOLD_MID  = 0.50  # Between → upgrade to SUSPICIOUS (MEDIUM risk)
    THRESHOLD_HIGH = 0.70  # Above → force ADVERSARIAL (HIGH risk)
```

---

### 🗺️ 7. Change XAI Layer (Grad-CAM Target) — `pipeline.py` / `utils/visualize.py`

By default, Grad-CAM hooks into `layer4` (deepest ResNet block — captures high-level features).

To visualize earlier layers (shows lower-level texture/edge features):
```python
# In pipeline.py, change:
fig = apply_gradcam(model, image_path, target_layer="layer3")  # earlier layer
```

| Layer | What it shows |
|-------|--------------|
| `layer4` | High-level semantic features (default — best for explainability) |
| `layer3` | Mid-level patterns |
| `layer2` | Texture & edges |
| `layer1` | Low-level color/gradient artifacts |

---

## ⚡ Quick Command Reference

```powershell
# Activate environment
venv\Scripts\activate

# Verify GPU
venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"

# Phase 1: Train on clean data
venv\Scripts\python.exe train.py --mode clean --data_dir data/clean --epochs 30

# Phase 2: Generate adversarial examples  
venv\Scripts\python.exe train.py --mode generate --data_dir data/clean --checkpoint checkpoints/best_model.pth

# Phase 3: Fine-tune as adversarial detector
venv\Scripts\python.exe train.py --mode adversarial --csv data/labels.csv --checkpoint checkpoints/best_model.pth --epochs 20

# Run full pipeline on test image
venv\Scripts\python.exe pipeline.py --image images.jpg --checkpoint checkpoints/best_model.pth

# Evaluate CNN only (no LLM needed)
venv\Scripts\python.exe evaluate.py --image images.jpg --checkpoint checkpoints/best_model.pth
```

---

## 🔑 Key Files Summary

| What to change | Edit this file |
|----------------|---------------|
| Hyperparameters, epochs, LR, batch size | `model/config.py` |
| CNN backbone (ResNet, EfficientNet, etc.) | `model/cnn_model.py` |
| Dataset size / identity count | `extract_casia.py` |
| Attack types (add CW, DeepFool, etc.) | `model/adversarial.py` |
| LLM provider / model / API key | `.env` only |
| Threshold weights / risk boundaries | `model/threshold.py` |
| Grad-CAM target layer | `pipeline.py` |
| Add a new LLM provider | `agents/llm_factory.py` |

---

## 🧠 Theory & Presentation FAQ
*Key concepts to explain during your project defense.*

### 1. Why use 4 Parameters for the Dynamic Threshold?
We use exactly 4 parameters (Confidence, Anomaly, Drift, Quality) because they perfectly cover the attack surface without adding unnecessary latency:
*   **Confidence (35%):** Evaluates the neural output, but is prone to adversarial overconfidence.
*   **Anomaly (30%):** Evaluates the mathematical distribution to see if the image was pushed off the "normal face" manifold.
*   **Embedding Drift (20%):** Evaluates identity features via cosine distance to catch attacks that alter the face's core mathematical identity.
*   **Image Quality/Laplacian (15%):** Evaluates the physical optics. Adversarial noise often introduces high-frequency artifacts (sharpness/blurriness) that act as a real-world sanity check.
*Adding more parameters (like FFT) would cause diminishing returns and increase processing latency beyond acceptable real-time limits.*

### 2. Why train ResNet-18 for 30 Epochs instead of 5-10?
While a simple ResNet-18 classifier can converge in 5-10 epochs, our architecture requires more:
*   **Multi-Task Learning (Joint Loss):** Our model computes both CrossEntropy (for the classifier) and BCELoss (for the Anomaly Head) simultaneously. Finding a global optimum that satisfies all distinct heads requires extended backpropagation.
*   **LR Polishing:** The Learning Rate drops at Epoch 10 and 20. The final epochs allow the model to "fine-tune" at a very low learning rate, squeezing out the high-precision embeddings needed for our drift detection.
*   **Early Stopping:** 30 is just a ceiling limit. The `trainer.py` uses Early Stopping (Patience = 7), meaning if the model stops improving, training will automatically halt to prevent overfitting.

### 3. What does "Pretrained on ImageNet" mean?
Instead of initializing the CNN with random "junk" weights (like a newborn baby learning to see), we load weights pretrained on ImageNet (1.4 million images). 
*   **The Analogy:** It's like hiring a college graduate. They already understand edges, textures, and shapes. We only need to give them a "crash course" (Phase 1 training) to apply that knowledge specifically to human faces and adversarial noise.

### 4. If the CNN is "fooled", how does it trigger the defense?
This is the core innovation of the project. We do **not** rely purely on the CNN's classification probability.
*   Even if the CNN is completely fooled into outputting "99% NORMAL", the mathematical structure of the adversarial face is unnatural. 
*   Because of this, the **Anomaly Head** and **Embedding Drift** scores will spike.
*   The `DynamicThresholdEngine` calculates the final equation `T`. Because the Anomaly/Drift multipliers are so high, `T` easily crosses the `0.25` safety boundary, overriding the "Fooled" CNN and waking up the LLM Agents.

### 5. Full-Stack Web App Workflow
If deployed as a real product, here is the journey of an image:
1.  **Frontend (Client):** User uploads an image via a web browser. The frontend securely POSTs this image to the backend.
2.  **Backend (Factory):** A Flask/FastAPI server receives the image and runs `pipeline.py`.
3.  **Processing (Math):** The ResNet-18 calculates Confidence, Anomaly, and Drift. The Threshold Engine calculates `T`.
4.  **Boardroom (LLM Agents):** If `T` crosses the limit, the backend calls the Groq/OpenAI APIs. The Advocates debate, and the Judge issues a verdict.
5.  **Packaging & Display:** The backend generates the Grad-CAM heatmap, bundles it with the Judge's JSON report, and sends it to the Frontend, which renders a dashboard showing the user why they were rejected.
