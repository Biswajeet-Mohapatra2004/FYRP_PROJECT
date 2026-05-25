# 🗺️ FYRP: Architecture & Phase Mapping
**A Categorized Guide to the Project Files and Folders**

To help you understand the architecture of the project, this guide groups all files and folders by the **chronological phases** of the workflow.

---

### 🟢 Phase 0: Environment & Dataset Preparation
*The phase where you get the raw data ready for the neural network.*

*   **`extract_casia.py`** & **`setup_lfw.py`**: Used to process the raw dataset (which is in a proprietary RecordIO format) and extract a smaller, manageable subset (10,000 images / 2,000 identities from CASIA, and 10,000 images from LFW "in-the-wild" dataset) into standard `.jpg` files.
*   **`data/clean/` and `data/lfw_clean/` (Folders)**: The output of this phase. It holds your clean, unmanipulated face images organized by identity subfolders.
*   **`requirements.txt`** & **`setup_venv.bat`**: Used one time to set up your Python environment and install PyTorch/LangChain/facenet-pytorch.

---

### 🧠 Phase 1: Base Model Training (Clean Data)
*The phase where the ResNet-18 learns what a normal human face looks like.*

*   **`train.py`** *(Run with `--mode clean`)*: The main script that triggers the training loop.
*   **`model/dataset.py`**: Loads the images from `data/clean/`, resizes them to 224x224, and feeds them to the GPU.
*   **`model/cnn_model.py`**: Contains the actual ResNet-18 architecture with your 3 custom heads (Classification, Anomaly, Embedding).
*   **`model/trainer.py`**: Handles the actual math of training (calculating Joint Loss [CrossEntropy + BCE], stepping the optimizer, early stopping).
*   **`checkpoints/best_model.pth` (File)**: The output of this phase. The saved "brain" or weights of the clean model.

---

### ⚔️ Phase 2: Adversarial Attack Generation
*The phase where you attack your own Phase 1 model to generate a threat dataset.*

*   **`train.py`** *(Run with `--mode generate`)*: Triggers the attack generation.
*   **`model/adversarial.py`**: Contains the raw mathematical formulas for **FGSM** and **PGD** attacks. It uses the `best_model.pth` from Phase 1, figures out how to trick it, and alters the pixels of the clean images.
*   **`data/adversarial/` (Folder)**: The output of this phase. It stores the newly generated hacked/noisy images.
*   **`data/labels.csv` (File)**: A master list created during this phase that maps every image to its true label (Normal vs. Adversarial).

---

### 🏋️ Phase 3: Adversarial Fine-Tuning
*The phase where the model learns to spot the attacks you just generated.*

*   **`train.py`** *(Run with `--mode adversarial`)*: Triggers the final training loop.
*   It uses **`model/dataset.py`** to read `labels.csv`, mixing the `data/clean/` and `data/adversarial/` folders together.
*   It updates **`checkpoints/best_model.pth`** so the model is now a hardened, adversarial-aware detector.

---

### 🤖 Phase 4: Multi-Agent Inference & Explainability
*The phase where the system goes live, evaluating a new image using the LLM Agents and Grad-CAM.*

*   **`pipeline.py`**: The "conductor" script. You give it an image, and it runs the entire sequence below.
*   **`evaluate.py`**: A smaller version of the pipeline that just runs the CNN without the LLMs (used for quick batch testing).
*   **`model/threshold.py`**: Takes the CNN outputs and calculates the 4-factor **Dynamic Threshold** (Confidence, Anomaly, Drift, Quality).
*   **`agents/advocate.py`**: The LLM scripts that act as the Proponent (defending the image) and Opponent (attacking the image).
*   **`agents/judge.py`**: The LLM script that reads the advocates' debate and issues the final verdict.
*   **`utils/visualize.py`**: Generates the **Grad-CAM** heatmap showing exactly which pixels were altered.
*   **`generate_test_attacks.py`**: A standalone CLI tool to instantly generate FGSM/PGD adversarial attacks on a single clean image to test the model dynamically.
*   **`results/` (Folder)**: The output of this phase. It stores the final JSON reports and the Grad-CAM `.png` heatmap images.

---

### ⚙️ Core Infrastructure (Used Everywhere)
*Files that constantly run in the background regardless of the phase.*

*   **`model/config.py`**: The "Control Center". Every phase looks at this file to know the batch size, learning rate, threshold weights, and image dimensions.
*   **`.env`**: Stores your API keys (like Groq or OpenAI) so the Agents in Phase 4 can talk to the internet.
*   **`agents/llm_factory.py`**: The router that connects your code to the LLM provider specified in `.env`.
*   **`utils/preprocessing.py`**: Ensures every image (whether training or inference) is perfectly aligned and cropped using **MTCNN**, and then normalized with the exact same ImageNet math.
*   **`fyrp_comprehensive_walkthrough.ipynb`**: The interactive showcase file that runs pieces of all the phases above for presentation purposes.

---

### 📈 Phase 5: Presentation & Visualization
*Scripts exclusively designed to generate charts and graphics for your project defense/slides.*

*   **`generate_presentation_visuals.py`**: The master script that generates all project defense visuals.
*   **`results/1_accuracy_comparison.png`**: The 5-bar chart showing baseline vulnerability vs framework robustness.
*   **`results/2_framework_defense_comparison.png`**: A side-by-side grid of Clean vs FGSM vs PGD showcasing the framework's detection capabilities.
*   **`results/3_identity_attack_demo.png`**: A 4x3 conceptual grid illustrating how adversarial attacks force identity misclassification in standard networks.
*   **`generate_xai_dashboard.py`**: Generates the Unified XAI Dashboard.
*   **`results/unified_xai_dashboard.png`**: Combines the Spatial XAI (Grad-CAM heatmap) with the Mathematical XAI (the 4 parameters) to provide a complete, technical explanation for the audience.
