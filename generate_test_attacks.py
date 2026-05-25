import argparse
import os
import torch
from torchvision.utils import save_image

from model.cnn_model import AdversarialFaceDetector
from model.adversarial import FGSM, PGD
from model.config import Config
from utils.preprocessing import preprocess_image, denormalize

def generate_attacks(image_path, model_path="checkpoints/best_model.pth",
                     fgsm_epsilon=None, pgd_epsilon=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    fgsm_eps = fgsm_epsilon if fgsm_epsilon is not None else Config.FGSM_EPSILON
    pgd_eps  = pgd_epsilon  if pgd_epsilon  is not None else Config.PGD_EPSILON
    print(f"Attack strength: FGSM ε={fgsm_eps}  PGD ε={pgd_eps}")
    print(f"  (pixel change ≈ {fgsm_eps*255:.1f}/255 per channel — "
          f"{'imperceptible' if fgsm_eps <= 0.01 else 'visible' if fgsm_eps >= 0.03 else 'subtle'})")

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

    # Initialize attackers with the chosen epsilon
    fgsm_attacker = FGSM(model, epsilon=fgsm_eps)
    pgd_attacker  = PGD(model,  epsilon=pgd_eps, alpha=pgd_eps / 4)

    # Generate FGSM
    print("Generating FGSM attack...")
    fgsm_adv = fgsm_attacker.perturb(images, labels)
    fgsm_adv_denorm = denormalize(fgsm_adv[0])

    # Generate PGD
    print("Generating PGD attack...")
    pgd_adv = pgd_attacker.perturb(images, labels)
    pgd_adv_denorm = denormalize(pgd_adv[0])

    # Save images — encode epsilon in filename for traceability
    base, ext = os.path.splitext(image_path)
    eps_tag = f"_eps{fgsm_eps:.4f}".rstrip('0').rstrip('.')
    fgsm_path = f"{base}_fgsm{eps_tag}{ext}"
    pgd_path  = f"{base}_pgd{eps_tag}{ext}"

    save_image(fgsm_adv_denorm, fgsm_path)
    save_image(pgd_adv_denorm, pgd_path)

    print("\n[Done] Adversarial images generated successfully!")
    print(f"  - FGSM saved to: {fgsm_path}")
    print(f"  - PGD  saved to: {pgd_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate FGSM and PGD adversarial images for a single image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Epsilon guide:
  0.001–0.005  imperceptible (recommended for demos)
  0.010–0.020  subtle noise visible on inspection
  0.030        standard benchmark value (visibly distorted)

Examples:
  python generate_test_attacks.py tony_stark.jpg --epsilon 0.005
  python generate_test_attacks.py tony_stark.jpg --epsilon 0.01
  python generate_test_attacks.py tony_stark.jpg               # uses config default (0.03)
""")
    parser.add_argument("image_path", type=str, help="Path to the clean image file.")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth",
                        help="Path to model checkpoint.")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Perturbation budget for both FGSM and PGD (overrides config). "
                             "E.g. 0.005 for imperceptible, 0.03 for standard benchmark.")
    parser.add_argument("--fgsm_epsilon", type=float, default=None,
                        help="Override epsilon for FGSM only.")
    parser.add_argument("--pgd_epsilon", type=float, default=None,
                        help="Override epsilon for PGD only.")

    args = parser.parse_args()

    # --epsilon sets both; individual flags override per-attack
    fgsm_eps = args.fgsm_epsilon or args.epsilon
    pgd_eps  = args.pgd_epsilon  or args.epsilon

    generate_attacks(args.image_path, args.model,
                     fgsm_epsilon=fgsm_eps, pgd_epsilon=pgd_eps)
