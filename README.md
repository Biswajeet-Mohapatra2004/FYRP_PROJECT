# Explainable Multi-Agent AI-Based Adversarial Threat Detection for Facial Biometric Systems

> **Final Year Research Project (FYRP)**  
> Training: Google Colab T4 · Inference: CPU

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
├── train.py                 — Training CLI (modes: clean / adversarial / generate)
│                              --epsilon flag for multi-epsilon data generation
├── evaluate.py              — Inference CLI + FAR/FRR biometric metrics
├── pipeline.py              — End-to-end orchestrator (conditional debate gate)
├── build_reference_gallery.py — One-time: extract 512-d embeddings from clean images
├── generate_test_attacks.py — FGSM/PGD single-image attacker (PNG output)
├── create_colab_zip.py      — Produces source-only zip (~1.7 MB) for Colab upload
├── download_casia.py        — Downloads CASIA-WebFace via kagglehub
├── extract_casia.py         — Converts .rec/.idx → JPEG files in data/clean/
├── setup_lfw.py             — Extracts LFW images for in-the-wild balance
├── requirements.txt
├── .env.example             — Copy to .env and add your LLM API key
│
├── model/
│   ├── config.py            — All hyperparameters & paths (single source of truth)
│   ├── cnn_model.py         — AdversarialFaceDetector (ResNet-18, 3 heads)
│   ├── dataset.py           — FaceDataset, AdversarialDataset, DatasetBuilder
│   ├── adversarial.py       — FGSM, PGD (pixel-space correct) + DatasetGenerator
│   ├── threshold.py         — DynamicThresholdEngine (4-factor adaptive threshold)
│   └── trainer.py           — Full training loop (early stopping, LR scheduler)
│
├── agents/
│   ├── advocate.py          — AdvocateAgent (Proponent + Opponent) via LangChain
│   ├── judge.py             — JudgeAgent (LLM-as-a-Judge, can override advocates)
│   └── llm_factory.py      — Provider-agnostic LLM instantiation (Groq / OpenAI)
│
├── frontend/
│   ├── index.html           — Neon research dashboard UI
│   ├── styles.css           — Neon glows, particle background, animations
│   └── script.js            — Particle canvas, step loader, live metric display
│
└── utils/
    ├── preprocessing.py     — MTCNN face alignment + image normalization
    └── visualize.py         — Grad-CAM (abs fallback for FGSM), training curves
```

---

## ⚙️ Setup (Local Inference Machine)

### 1. Clone & enter repo
```bash
git clone <your-github-repo-url>
cd FYRP_codebase
```

### 2. Create venv & install dependencies

**Option A: Automated (Windows)**
```bash
.\setup_venv.bat
```

**Option B: Manual**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install "torchattacks>=3.5.1" --no-deps  # bypasses requests version conflict
```

### 3. Set up LLM API key
```bash
copy .env.example .env
# Edit .env — set GROQ_API_KEY or OPENAI_API_KEY
```

### 4. Place checkpoints
Download `best_model.pth` and `reference_gallery.pt` into `checkpoints/`.  
(Pre-trained on Colab T4, multi-epsilon adversarial data, 100% test accuracy.)

### 5. Start the system
```bash
venv\Scripts\python.exe app.py         # backend on http://localhost:8001
# Open frontend/index.html in browser
```

---

## 🚀 Training Workflow (Google Colab T4 — Recommended)

Training is done on Google Colab (free T4 GPU). Use `create_colab_zip.py` to
package the source code, upload to Drive, then run the cells below.

```bash
# Step 0 — create source-only zip (~1.7 MB)
python create_colab_zip.py
# Upload FYRP_code.zip + checkpoints/ + lfw_clean/ to Google Drive
```

### Phase 1 — Train CNN on clean faces
```bash
python train.py --mode clean --data_dir data/lfw_clean --epochs 30 --batch_size 32
# Output: checkpoints/best_model.pth
```

### Phase 2a — Generate adversarial examples at multiple epsilons
For maximum robustness, generate data at 4 epsilon values and combine:
```bash
for eps in 0.01 0.02 0.03 0.05; do
    python train.py --mode generate \
        --data_dir data/lfw_clean \
        --checkpoint checkpoints/best_model.pth \
        --epsilon $eps --batch_size 32
done
# Output per epsilon: data/labels_eps<e>.csv + data/adversarial/fgsm_eps<e>/
```

Then combine CSVs:
```python
import pandas as pd, glob
dfs = [pd.read_csv(f) for f in glob.glob("data/labels_eps*.csv")]
pd.concat(dfs).sample(frac=1).to_csv("data/labels_combined.csv", index=False)
```

### Phase 2b — Fine-tune as 3-class adversarial detector
```bash
python train.py --mode adversarial \
    --csv data/labels_combined.csv \
    --checkpoint checkpoints/best_model.pth \
    --epochs 15 --lr 0.00001 --batch_size 32
# Output: checkpoints/best_model.pth (updated, ~100% accuracy by epoch 2)
```

### Phase 3 — Build reference gallery
```bash
python build_reference_gallery.py \
    --clean_dir data/lfw_clean --n 500
# Output: checkpoints/reference_gallery.pt
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
```bash
python generate_test_attacks.py my_selfie.jpg --epsilon 0.03
# Output: my_selfie_fgsm_eps0.03.png and my_selfie_pgd_eps0.03.png
# Note: Always PNG — JPEG compression destroys PGD's structured perturbation
```

**Epsilon guide:**

| Epsilon | Pixel change | Perception | Detection |
|---------|-------------|------------|-----------|
| 0.01 | 2.5/255 | Invisible | Marginal |
| 0.02 | 5.1/255 | Invisible | Good |
| **0.03** | **7.6/255** | **Borderline** | **Strong — use for demo** |
| 0.05 | 12.7/255 | Slight noise | Very strong |

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
| DataLoader hangs on Windows | `NUM_WORKERS = 0` in `model/config.py` (already set) |
| `API_KEY not set` | Copy `.env.example` → `.env` and add your key |
| PGD image → LEGITIMATE | Use PNG not JPEG. JPEG compression destroys the perturbation. |
| Drift shows 0.0000 | Threshold uses P(normal)-based drift. If still 0, gallery may not have loaded — check logs for "Loaded reference gallery". |
| FGSM heatmap is solid blue | Fixed in visualize.py (abs fallback). Restart server to pick up change. |
| Colab: `PIL._util` ImportError | Run `pip install "Pillow>=10.0.0"` and restart runtime. |
| Colab: zip has flat backslash-named files | Run the path-reconstruction cell (replaces `\\` with `/` in filenames). |
| Corrupt image warnings | Expected — ~0.7% of CASIA-WebFace. Handled with try/except in dataset.py. |
