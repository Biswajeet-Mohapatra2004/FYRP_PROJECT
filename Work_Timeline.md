# 📅 FYRP Project Work Timeline
**Explainable Multi-Agent AI-Based Adversarial Threat Detection for Facial Biometric Systems**

This document serves as a chronological record of the project's development, highlighting the goals, challenges encountered, and the technical solutions implemented to achieve the final robust architecture.

---

## Phase 1: Environment Setup & Dataset Acquisition
**Goal:** Establish the working environment, download the primary dataset, and prepare it for PyTorch.

*   **Action:** Created `setup_venv.bat` and `requirements.txt`. Built `download_casia.py` to fetch the CASIA-WebFace dataset from Kaggle.
*   **Encountered Challenge:** The CASIA dataset was packaged in a proprietary binary format (`MXNet RecordIO` - `.rec`, `.idx`). PyTorch cannot read this natively.
*   **Resolution:** Engineered `extract_casia.py` to manually parse the binary format, skip the 24-byte headers, and extract the raw byte strings into standard `.jpg` images structured within `data/clean/<identity>` folders. Also created `create_subset.py` to allow rapid prototyping on a 10K image subset.

## Phase 2: Core Architecture & Base Training
**Goal:** Build the primary ResNet-18 feature extractor and train it to recognize clean facial embeddings.

*   **Action:** Developed `model/cnn_model.py` (ResNet-18 with 3 custom heads: Classifier, Embedding, Anomaly) and `model/trainer.py`.
*   **Encountered Challenge 1:** Dataloader freezing and hanging indefinitely on the Windows OS environment.
*   **Resolution 1:** Diagnosed a multiprocessing issue specific to Windows. Enforced `NUM_WORKERS = 0` in `model/config.py` as a fallback, ensuring stable, synchronous data loading.
*   **Encountered Challenge 2:** Occasional corrupt images in the CASIA dataset causing training crashes.
*   **Resolution 2:** Implemented a robust `try/except` fallback loop in `model/dataset.py` that gracefully catches corrupted `PIL` unpickling errors and defaults to a zero-tensor.

## Phase 3: Adversarial Generation & Fine-Tuning
**Goal:** Generate mathematical adversarial threats (FGSM/PGD) and train the CNN to act as a 3-class defense system.

*   **Action:** Authored `model/adversarial.py` containing custom implementations of Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD). Executed Phase 1 training, then generated 20,000 attacked images based on the model's own gradients.
*   **Encountered Challenge:** Mixing clean and generated adversarial data dynamically during training could lead to memory leaks and dataset indexing issues.
*   **Resolution:** Opted for a static-generation approach. We saved all attacked images physically to `data/adversarial/` and created a centralized `labels.csv` manifest. The Phase 2 dataloader simply reads this CSV, allowing perfectly shuffled, deterministic fine-tuning.

## Phase 4: Multi-Agent Logic & Explainability (XAI)
**Goal:** Replace the industry-standard "fixed threshold" with a dynamic equation and build the LLM Agent Debate system.

*   **Action:** Built `model/threshold.py` containing the `DynamicThresholdEngine` (calculating Risk `T` based on Confidence, Anomaly, Drift, and Quality). Integrated LangChain in `agents/` to create the Advocate (Proponent/Opponent) and Judge agents. Added Grad-CAM in `utils/visualize.py`.
*   **Encountered Challenge:** LLM API integration was rigid; switching from Groq to OpenAI required deep code refactoring.
*   **Resolution:** Architected the `llm_factory.py` design pattern. This centralized the LLM instantiation to a single function reading from `.env`, making the entire agentic backend completely provider-agnostic.

## Phase 5: Web UI & Artifact Management
**Goal:** Build a cohesive FastAPI frontend to present the XAI dashboard to end users.

*   **Action:** Developed `app.py` and `dashboard_generator.py` to stitch the Grad-CAM visuals, threshold graphs, and LLM text into a single unified dashboard `.png`.
*   **Encountered Challenge:** The pipeline output was messy, cluttering the results folder with `_gradcam.png`, `_pure_overlay.png`, and raw JSON files every time an image was tested.
*   **Resolution:** Updated the pipeline execution logic to retain only the final beautifully stitched `_dashboard.png`, automatically wiping intermediate artifacts to keep the server directory pristine.

## Phase 6: The "In-The-Wild" Domain Gap (Final Polish)
**Goal:** Ensure the model works perfectly on legitimate user phone selfies, outside of strict laboratory dataset conditions.

*   **Encountered Challenge 1:** The model flagged *every* uploaded phone selfie as "Suspicious." The background (walls, shoulders) was vastly different from the tightly cropped CASIA dataset, triggering false positives.
*   **Resolution 1:** Integrated **MTCNN** into `utils/preprocessing.py`. This enforced strict facial detection and cropping *before* the tensor entered the CNN, successfully normalizing the geometric domain gap.
*   **Encountered Challenge 2 (Critical Architectural Bug):** Even with MTCNN, the Anomaly Score was violently spiking on clean faces. Investigation revealed that the `anomaly_head` inside the CNN was essentially a random number generator because it was entirely disconnected from the backpropagation loss loop in `trainer.py`.
*   **Resolution 2 (Mathematical Fix):** Completely rewrote the training loop to implement a **Joint Loss Function**. We added `nn.BCELoss` specifically targeting the Anomaly Head, and combined it with the classification `CrossEntropyLoss`.
*   **Resolution 3 (Data Equalization):** Realizing that the model had only learned the specific texture of CASIA images, we introduced the **LFW (Labeled Faces in the Wild)** dataset. We downloaded 10,000 clean LFW selfies, generated 20,000 LFW-based attacks, and re-ran the Joint Loss fine-tuning on a perfectly balanced 1:1 CASIA/LFW ratio. This allowed the anomaly head to mathematically map the structural variance of consumer-grade selfies, eliminating the false positives.

## Phase 8: Session 2 — Comprehensive Bug Fixes & Hardening (2026-05-26)
**Goal:** Fix all known issues in the pipeline without retraining. Make the threshold engine, agent parsing, and web UI work correctly.

*   **Environment:** Recreated venv on new Windows 11 machine (Python 3.11). Fixed port 8000 → 8001 (conflict). Fixed hardcoded dashboard URL in app.py.

*   **Fix 1 — Threshold boundaries too high (CRITICAL):** `THRESHOLD_LOW=0.35` meant T=0.26 (adversarial images) was always below the boundary → threshold engine never upgraded any label. Lowered to LOW=0.20 / MID=0.40 / HIGH=0.65. Fixed `_adjust_label()` so suspicious upgrades to adversarial in the MID–HIGH range.

*   **Fix 2 — Image quality always zero:** The Laplacian variance was computed on the normalized tensor (values ~[-2,2]) instead of the pixel image. Variance was ~0 for all images, making quality_factor always 0.5. Fixed by adding ImageNet denormalization before computing Laplacian.

*   **Fix 3 — Threshold weights hardcoded:** Moved W_CONFIDENCE/W_ANOMALY/W_DRIFT/W_QUALITY from class constants to constructor keyword arguments. Added threshold_low/mid/high as constructor overrides. Weights can now be tuned per-experiment without code changes.

*   **Fix 4 — Embedding drift always 0.0:** Reference gallery was never loaded (None), zeroing the 0.20*drift term. Created `build_reference_gallery.py` to extract 512-d embeddings from clean images. Downloaded LFW dataset (13,233 images), built 500-embedding gallery. Updated `pipeline.py` to auto-load gallery from checkpoints/reference_gallery.pt.

*   **Fix 5 — Unconditional debate gate:** Every image triggered expensive LLM calls regardless of how clear-cut the result was. Added conditional gate: T<0.20 → fast-path LEGITIMATE, T>0.65 → fast-path ADVERSARIAL, ambiguous range → full debate.

*   **Fix 6 — Judge missing ground truth:** Judge only received advocate arguments; if an advocate missed key evidence the Judge couldn't recover it. Updated `judge.verdict()` to accept raw CNN output as a third argument. Updated Judge system prompt threshold guideline to match new boundaries.

*   **Fix 7 — Fragile response parsing:** Both advocate.py and judge.py used `startswith()` string matching, silently defaulting when LLM output had minor formatting variations. Rewrote both parsers to use `stripped.partition(":")` with per-field try/except and logger warnings on defaults.

*   **Fix 8 — Missing FAR/FRR metrics:** evaluate.py had no biometric security metrics. Added FAR (adversarial wrongly classified as normal) and FRR (normal wrongly classified as adversarial) using sklearn confusion_matrix.

*   **Fix 9 — UI showing all zeros:** frontend/script.js was reading `threshold_details.factors.confidence_factor` (inverted factor, 0 when confidence=1.0) instead of the raw scores. Fixed to read `confidence_score`, `anomaly_score`, `embedding_drift`, `image_quality_score` directly.

*   **Critical Bug — Adversarial attack pixel-space error:** FGSM and PGD applied `epsilon * sign(grad)` to normalized tensors then clamped to `[0,1]`. Since normalized images span ~[-2.5, 2.5], the clamp clipped ~50% of pixels to zero regardless of epsilon — producing grey-blob distortions and making epsilon changes have no visible effect. Fixed both FGSM and PGD to: denormalize → apply epsilon perturbation in pixel [0,1] space → clamp → re-normalize.

*   **Pending Issue — Model cannot detect correct adversarial examples:** After the pixel-space fix, model gives LEGITIMATE for epsilon=0.03 adversarial images. Root cause: the model was trained on the old buggy attacks (effective pixel perturbation ~0.13), which produced grey-blob artifacts. The model learned those specific distortions, not real imperceptible perturbations. Requires re-generating adversarial data with fixed code and fine-tuning for 10 epochs (see context.txt Section 4).

## Phase 7: Completion & Handoff
**Goal:** Finalize documentation and testing capabilities.

*   **Action:** Wrote `generate_test_attacks.py` to allow users to generate standalone FGSM/PGD attacks from the terminal. Updated all core architectural documents (`README.md`, `FYRP_ARCHITECTURE_MAPPING.md`, `FYRP_PROJECT_GUIDE.md`, and `context.txt`) to accurately reflect the final, fully optimized state of the framework.
*   **Encountered Challenge:** A severe "dependency hell" `ResolutionImpossible` error occurred during fresh installations. `torchattacks 3.5.1` demanded an outdated `requests~=2.25.1` (from 2020), which catastrophically conflicted with modern `langchain` and `tiktoken` dependencies requiring `requests>=2.26.0`.
*   **Resolution:** Engineered a bypass in `setup_venv.bat`. Removed `torchattacks` from the primary `requirements.txt` to allow the modern LangChain/OpenAI stack to resolve seamlessly, then implemented a post-installation hook `pip install torchattacks>=3.5.1 --no-deps` to force-install the attack library without dragging down the environment's network requests handler.
