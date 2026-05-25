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

## Phase 7: Completion & Handoff
**Goal:** Finalize documentation and testing capabilities.

*   **Action:** Wrote `generate_test_attacks.py` to allow users to generate standalone FGSM/PGD attacks from the terminal. Updated all core architectural documents (`README.md`, `FYRP_ARCHITECTURE_MAPPING.md`, `FYRP_PROJECT_GUIDE.md`, and `context.txt`) to accurately reflect the final, fully optimized state of the framework.
*   **Encountered Challenge:** A severe "dependency hell" `ResolutionImpossible` error occurred during fresh installations. `torchattacks 3.5.1` demanded an outdated `requests~=2.25.1` (from 2020), which catastrophically conflicted with modern `langchain` and `tiktoken` dependencies requiring `requests>=2.26.0`.
*   **Resolution:** Engineered a bypass in `setup_venv.bat`. Removed `torchattacks` from the primary `requirements.txt` to allow the modern LangChain/OpenAI stack to resolve seamlessly, then implemented a post-installation hook `pip install torchattacks>=3.5.1 --no-deps` to force-install the attack library without dragging down the environment's network requests handler.
