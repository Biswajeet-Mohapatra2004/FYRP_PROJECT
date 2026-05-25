# Explainable Multi-Agent AI-Based Adversarial Threat Detection for Facial Biometric Systems

> **Final Year Research Project (FYRP)**  
> GPU Machine: RTX 3060 (12 GB VRAM)

---

## 📖 Overview

This project detects adversarial attacks on facial biometric systems using a three-stage pipeline:

1. **CNN Detection** — ResNet-18 classifies faces as `normal`, `adversarial`, or `suspicious`
2. **Dynamic Threshold Engine** — replaces fixed thresholds with a 4-factor context-aware score
3. **Multi-Agent Debate** — Proponent + Opponent LLM agents argue; a Judge LLM issues the final verdict
4. **Grad-CAM XAI** — highlights *where* on the face the adversarial perturbation was detected

### 3 Core Novelties
| # | Novelty | Description |
|---|---------|-------------|
| 1 | **Dynamic Threshold Engine** | `T = 0.35*(1-conf) + 0.30*anomaly + 0.20*drift + 0.15*(1-quality)` |
| 2 | **Advocate–Judge Debate (MAD)** | LLM-as-a-Judge prevents single-agent hallucination propagation |
| 3 | **Grad-CAM XAI Layer** | Shows *where* the attack is on the face |

---

## 🗂️ Project Structure

```
FYRP_codebase/
├── train.py              — Training CLI (modes: clean / adversarial / generate)
├── evaluate.py           — Inference CLI (single image or batch CSV)
├── pipeline.py           — End-to-end orchestrator (all components chained)
├── download_casia.py     — Downloads CASIA-WebFace via kagglehub
├── extract_casia.py      — Converts .rec/.idx → JPEG files in data/clean/
├── setup_lfw.py          — Extracts 10,000 LFW images for in-the-wild feature balance
├── generate_test_attacks.py — CLI to test FGSM/PGD generation on a single image
├── create_subset.py      — CPU-friendly 500-identity subset for quick tests
├── requirements.txt
├── .env.example          — Copy to .env and add your OpenAI key
│
├── model/
│   ├── config.py         — All hyperparameters & paths (single source of truth)
│   ├── cnn_model.py      — AdversarialFaceDetector (ResNet-18, 3 heads)
│   ├── dataset.py        — FaceDataset, AdversarialDataset, DatasetBuilder
│   ├── adversarial.py    — FGSM, PGD attacks + AdversarialDatasetGenerator
│   ├── threshold.py      — DynamicThresholdEngine (4-factor adaptive threshold)
│   └── trainer.py        — Full training loop (early stopping, LR scheduler)
│
├── agents/
│   ├── advocate.py       — AdvocateAgent (Proponent + Opponent) via LangChain
│   └── judge.py          — JudgeAgent (LLM-as-a-Judge, can override advocates)
│
└── utils/
    ├── preprocessing.py  — MTCNN face alignment, preprocess_image(), format_features_for_agent()
    └── visualize.py      — Grad-CAM, confusion matrix, training curves
```

---

## ⚙️ Setup (GPU Machine — RTX 3060)

### 1. Clone & enter repo
```bash
git clone <your-github-repo-url>
cd FYRP_codebase
```

### 2. Create venv & install dependencies

**Option A: Use the automated script (Windows Recommended)**
```bash
.\setup_venv.bat
```

**Option B: Manual setup**
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install "torchattacks>=3.5.1" --no-deps  # Important: bypasses 'requests' version conflict
```

### 3. Install PyTorch with CUDA (RTX 3060 → CUDA 12.1)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Verify GPU is detected
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected: True | NVIDIA GeForce RTX 3060
```

### 5. Set up Kaggle API credentials (for dataset download)
- Go to https://www.kaggle.com → Account → API → **Create New Token**
- Place `kaggle.json` at `C:\Users\<YourUsername>\.kaggle\kaggle.json`

### 6. Download & extract datasets (CASIA + LFW)
```bash
python download_casia.py    # ~2.7 GB, uses kagglehub
python extract_casia.py     # ~490K images → data/clean/
python -c "import setup_lfw; setup_lfw.setup_lfw_subset(10000)" # Extracts 10,000 LFW images
```
> Takes ~15–20 minutes total. Alternatively, copy `data/clean/` from the old machine.

### 7. Set up OpenAI API key
```bash
copy .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

---

## 🚀 Training Workflow

### Phase 1 — Train CNN on clean faces
```bash
python train.py --mode clean --data_dir data/clean --epochs 30 --batch_size 64
# Output: checkpoints/best_model.pth
# Time: ~5–10 min/epoch on RTX 3060
```

### Phase 2a — Generate adversarial examples (FGSM + PGD)
```bash
python train.py --mode generate \
    --data_dir data/clean \
    --checkpoint checkpoints/best_model.pth
# Output: data/labels.csv + data/adversarial/fgsm/ + data/adversarial/pgd/
```

### Phase 2b — Fine-tune as 3-class adversarial detector
```bash
python train.py --mode adversarial \
    --csv data/labels.csv \
    --checkpoint checkpoints/best_model.pth \
    --epochs 20 --batch_size 64
# Output: checkpoints/best_model.pth (updated)
```

---

## 🔍 Inference

### Single image via evaluate.py
```bash
python evaluate.py --image images.jpg --checkpoint checkpoints/best_model.pth
```

### Full pipeline (CNN + Threshold + Agents + Grad-CAM + Dashboard)
```bash
python pipeline.py \
    --image images.jpg \
    --checkpoint checkpoints/best_model.pth \
    --model gpt-4o
# Output: results/<image>_report.json + results/<image>_dashboard.png
```

### Standalone Attack Generator
To generate an adversarial image without running the pipeline:
```bash
python generate_test_attacks.py my_selfie.jpg
# Output: my_selfie_fgsm.jpg and my_selfie_pgd.jpg
```

---

## 🧠 Pipeline Architecture

```
Input Image
    │
    ▼
[1] Preprocessing (PIL, torchvision)
    │  → tensor (1, 3, 224, 224)
    ▼
[2] CNN Feature Extraction (ResNet-18)
    │  → prediction, confidence, anomaly_score, embedding (512-d)
    ▼
[3] Dynamic Threshold Engine
    │  T = 0.35*(1-conf) + 0.30*anomaly + 0.20*drift + 0.15*(1-quality)
    │  → final_label, risk_level, computed_threshold
    ▼
[4] Advocate Agents (LangChain + GPT-4o)
    │  ├─ Proponent argues: LEGITIMATE
    │  └─ Opponent argues:  ADVERSARIAL
    ▼
[5] Judge Agent (LLM-as-a-Judge)
    │  → final_decision, override, reasoning
    ▼
[6] Grad-CAM XAI Heatmap
    │  → results/<image>_gradcam.png
    ▼
Final Report (JSON)
    → results/<image>_report.json
```

---

## 📊 Output Report Format (`results/<image>_report.json`)

```json
{
  "image_path": "images.jpg",
  "cnn_prediction": "NORMAL",
  "final_decision": "SUSPICIOUS",
  "risk_level": "MEDIUM",
  "confidence": "medium",
  "reasoning": "...",
  "override": true,
  "override_reason": "Low confidence score warrants elevated caution.",
  "heatmap_path": "results/images_gradcam.png",
  "threshold_details": { ... },
  "advocate_pro": { ... },
  "advocate_opp": { ... },
  "elapsed_seconds": 12.4,
  "timestamp": "2026-04-29T08:30:00Z"
}
```

---

## 📚 Research Context

This project addresses the gap in:
> **Park et al. (2024)** — *"A Comprehensive Risk Analysis Method for Adversarial Attacks on Biometric Authentication Systems"*  
> They used **fixed thresholds** across 7 attack surfaces × 12 scenarios → "one-size-fits-all" problem.

**Our solution**: Dynamic threshold + agent-based reasoning + XAI explainability.

| Reference | Role in this project |
|-----------|---------------------|
| Park et al. (2024) | Gap identified → fixed thresholds |
| Verheyen et al. (2023) | Adaptive thresholds still fail (doppelganger) |
| Guo et al. (2025) | FGSM, PGD attack survey |
| Li et al. (2026) | PhishDebate: multi-agent in cybersecurity (proof) |
| Chen et al. (2026) | G-DMAD: MAD improves reasoning |
| Acharya et al. (2025) | Agentic AI survey (justifies architecture) |
| Apap et al. (2024) | Explainable biometrics (justifies Grad-CAM) |

---

## 🔧 Troubleshooting

| Issue | Fix |
|-------|-----|
| `torch.cuda.is_available()` returns `False` | Reinstall PyTorch with correct CUDA version: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| DataLoader hangs on Windows | Set `NUM_WORKERS = 0` in `model/config.py` |
| `OPENAI_API_KEY not set` | Copy `.env.example` to `.env` and add your key |
| `No images found` in data/clean/ | Run `python extract_casia.py` first |
| Corrupt image warnings | Expected — ~0.7% of CASIA-WebFace. Handled with try/except in dataset.py |
