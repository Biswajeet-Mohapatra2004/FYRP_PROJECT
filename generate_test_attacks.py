import argparse
import os
import torch
from torchvision.utils import save_image

from model.cnn_model import AdversarialFaceDetector
from model.adversarial import FGSM, PGD
from utils.preprocessing import preprocess_image, denormalize

def generate_attacks(image_path, model_path="checkpoints/best_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = AdversarialFaceDetector()
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device, weights_only=True)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Checkpoint not found at {model_path}. Using random weights.")
    
    model.to(device)
    model.eval()

    # Load and preprocess image
    print(f"Loading and preprocessing {image_path}...")
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    images = preprocess_image(image_path).to(device)
    labels = torch.tensor([0]).to(device)  # Assume ground truth is 0 (Clean)

    # Initialize attackers
    fgsm_attacker = FGSM(model)
    pgd_attacker = PGD(model)

    # Generate FGSM
    print("Generating FGSM attack...")
    fgsm_adv = fgsm_attacker.perturb(images, labels)
    fgsm_adv_denorm = denormalize(fgsm_adv[0])
    
    # Generate PGD
    print("Generating PGD attack...")
    pgd_adv = pgd_attacker.perturb(images, labels)
    pgd_adv_denorm = denormalize(pgd_adv[0])

    # Save images
    base, ext = os.path.splitext(image_path)
    fgsm_path = f"{base}_fgsm{ext}"
    pgd_path = f"{base}_pgd{ext}"

    save_image(fgsm_adv_denorm, fgsm_path)
    save_image(pgd_adv_denorm, pgd_path)

    print("\n[Done] Adversarial images generated successfully!")
    print(f"  - FGSM saved to: {fgsm_path}")
    print(f"  - PGD  saved to: {pgd_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FGSM and PGD adversarial images for a single image.")
    parser.add_argument("image_path", type=str, help="Path to the clean image file.")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth", help="Path to model checkpoint.")
    
    args = parser.parse_args()
    generate_attacks(args.image_path, args.model)
