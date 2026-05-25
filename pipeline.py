"""
Pipeline Orchestrator — End-to-End Adversarial Face Detection

Chains all components in the correct order:
  preprocess → CNN feature extraction → Dynamic Threshold Engine →
  Advocate Agents (Proponent + Opponent) → Judge Agent → Grad-CAM → Report

Usage:
    from pipeline import run_pipeline

    result = run_pipeline(
        image_path="path/to/face.jpg",
        checkpoint="checkpoints/best_model.pth",
        openai_api_key="sk-...",   # or set OPENAI_API_KEY env var
    )
    print(result["final_decision"])  # "LEGITIMATE" / "ADVERSARIAL" / "SUSPICIOUS"
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv

# ── Load .env if present ──
load_dotenv()

# ── Internal imports ──
from model.config import Config
from model.cnn_model import FeatureExtractor
from model.threshold import DynamicThresholdEngine
from utils.preprocessing import preprocess_image, format_features_for_agent
from utils.visualize import plot_gradcam
from agents.advocate import run_advocates, format_debate_for_judge
from agents.judge import JudgeAgent

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
#  Main pipeline function
# ────────────────────────────────────────────────────────────────────

def run_pipeline(
    image_path: str,
    checkpoint: str = None,
    openai_api_key: str = None,
    model_name: str = None,
    save_heatmap: bool = True,
    output_dir: str = None,
    reference_embeddings: list = None,
    verbose: bool = True,
) -> dict:
    """
    End-to-end adversarial detection pipeline for a single face image.

    Args:
        image_path          : path to the face image (str / Path)
        checkpoint          : path to trained model checkpoint (.pth)
                              If None, uses untrained model (for testing only)
        openai_api_key      : OpenAI API key (reads OPENAI_API_KEY env if None)
        model_name          : LLM model to use for agents (default: "gpt-4o")
        save_heatmap        : whether to save the Grad-CAM heatmap to disk
        output_dir          : directory for results (default: results/)
        reference_embeddings: list of clean reference embeddings for drift calc
        verbose             : log pipeline step timing

    Returns:
        dict with keys:
            image_path        : str
            cnn_prediction    : str
            final_decision    : str   ("LEGITIMATE" / "ADVERSARIAL" / "SUSPICIOUS")
            risk_level        : str   ("LOW" / "MEDIUM" / "HIGH")
            confidence        : str   ("high" / "medium" / "low")
            reasoning         : str
            override          : bool
            override_reason   : str
            heatmap_path      : str or None
            threshold_details : dict
            advocate_pro      : dict
            advocate_opp      : dict
            elapsed_seconds   : float
            timestamp         : str
    """
    pipeline_start = time.time()
    image_path = str(image_path)

    if verbose:
        logger.info(f"{'='*60}")
        logger.info(f"  FYRP PIPELINE — {Path(image_path).name}")
        logger.info(f"{'='*60}")

    # ── Output directory ──
    output_dir = output_dir or Config.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    # ─────────────────────────────────────────────
    # STEP 1: Preprocess image → tensor
    # ─────────────────────────────────────────────
    t0 = time.time()
    tensor = preprocess_image(image_path)         # (1, 3, 224, 224)
    if verbose:
        logger.info(f"[1/6] Preprocessing done ({time.time()-t0:.2f}s) — "
                    f"tensor shape: {tuple(tensor.shape)}")

    # ─────────────────────────────────────────────
    # STEP 2: CNN Feature Extraction
    # ─────────────────────────────────────────────
    t0 = time.time()
    extractor = FeatureExtractor(
        checkpoint_path=checkpoint,
        device=None,    # auto-detect CUDA / CPU
    )
    feature_dict = extractor.extract(tensor)
    cnn_str = format_features_for_agent(feature_dict)
    if verbose:
        logger.info(f"[2/6] CNN extraction done ({time.time()-t0:.2f}s) — "
                    f"prediction: {feature_dict['prediction'].upper()}, "
                    f"confidence: {feature_dict['confidence_score']:.4f}")

    # ─────────────────────────────────────────────
    # STEP 3: Dynamic Threshold Engine
    # ─────────────────────────────────────────────
    t0 = time.time()
    engine = DynamicThresholdEngine(reference_embeddings=reference_embeddings)
    threshold_decision = engine.compute(feature_dict, image_tensor=tensor)
    threshold_str = engine.format_for_agent(threshold_decision)
    if verbose:
        logger.info(f"[3/6] Threshold engine done ({time.time()-t0:.2f}s) — "
                    f"T={threshold_decision.computed_threshold:.4f}, "
                    f"final_label={threshold_decision.final_label.upper()}, "
                    f"risk={threshold_decision.risk_level}")

    # ─────────────────────────────────────────────
    # STEP 4: Advocate Agents — Proponent & Opponent
    # ─────────────────────────────────────────────
    t0 = time.time()
    pro_out, opp_out = run_advocates(
        cnn_output=cnn_str,
        threshold_output=threshold_str,
        model_name=model_name,
        api_key=openai_api_key,
    )
    debate_transcript = format_debate_for_judge(pro_out, opp_out)
    if verbose:
        logger.info(f"[4/6] Advocate debate done ({time.time()-t0:.2f}s)")
        logger.info(f"       Proponent → {pro_out.risk_assessment} | "
                    f"Opponent → {opp_out.risk_assessment}")

    # ─────────────────────────────────────────────
    # STEP 5: Judge Agent — Final Verdict
    # ─────────────────────────────────────────────
    t0 = time.time()
    judge = JudgeAgent(
        model_name=model_name,
        temperature=0.1,
        api_key=openai_api_key,
    )
    judge_verdict = judge.verdict(debate_transcript, threshold_str)
    if verbose:
        logger.info(f"[5/6] Judge verdict done ({time.time()-t0:.2f}s) — "
                    f"{judge_verdict.final_decision} | "
                    f"override={judge_verdict.override}")

    # ─────────────────────────────────────────────
    # STEP 6: Grad-CAM Heatmap (XAI)
    # ─────────────────────────────────────────────
    heatmap_path = None
    t0 = time.time()
    if save_heatmap:
        try:
            img_stem = Path(image_path).stem
            heatmap_filename = f"{img_stem}_gradcam.png"
            heatmap_path_full = os.path.join(output_dir, heatmap_filename)
            plot_gradcam(
                model=extractor.model,
                image_tensor=tensor,
                device=extractor.device,
                save_path=heatmap_path_full,
            )
            heatmap_path = heatmap_path_full
            if verbose:
                logger.info(f"[6/6] Grad-CAM saved to: {heatmap_path} "
                            f"({time.time()-t0:.2f}s)")
        except Exception as e:
            logger.warning(f"[6/6] Grad-CAM failed (non-critical): {e}")
    else:
        if verbose:
            logger.info("[6/6] Grad-CAM skipped (save_heatmap=False)")

    # ─────────────────────────────────────────────
    # Build final report
    # ─────────────────────────────────────────────
    elapsed = round(time.time() - pipeline_start, 2)
    from datetime import datetime
    timestamp = datetime.utcnow().isoformat() + "Z"

    report = {
        # ── Primary result ──
        "image_path":      image_path,
        "cnn_prediction":  feature_dict["prediction"].upper(),
        "final_decision":  judge_verdict.final_decision,
        "risk_level":      judge_verdict.risk_level,
        "confidence":      judge_verdict.confidence,
        "reasoning":       judge_verdict.reasoning,
        "override":        judge_verdict.override,
        "override_reason": judge_verdict.override_reason,
        "heatmap_path":    heatmap_path,
        # ── Technical details ──
        "threshold_details": threshold_decision.to_dict(),
        "advocate_pro": {
            "stance":          pro_out.stance,
            "risk_assessment": pro_out.risk_assessment,
            "confidence":      pro_out.confidence,
            "key_evidence":    pro_out.key_evidence,
            "argument":        pro_out.argument,
        },
        "advocate_opp": {
            "stance":          opp_out.stance,
            "risk_assessment": opp_out.risk_assessment,
            "confidence":      opp_out.confidence,
            "key_evidence":    opp_out.key_evidence,
            "argument":        opp_out.argument,
        },
        # ── Meta ──
        "elapsed_seconds": elapsed,
        "timestamp":       timestamp,
    }

    # Save JSON report alongside heatmap
    img_stem = Path(image_path).stem
    report_path = os.path.join(output_dir, f"{img_stem}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved: {report_path}")

    if verbose:
        _print_summary(report)

    return report


# ────────────────────────────────────────────────────────────────────
#  Helper: pretty-print summary to stdout
# ────────────────────────────────────────────────────────────────────

def _print_summary(report: dict):
    """Print a clean, human-readable summary of the pipeline result."""
    decision_map = {
        "LEGITIMATE":  "[OK]",
        "SUSPICIOUS":  "[!!]",
        "ADVERSARIAL": "[XX]",
    }
    risk_map = {"LOW": "[LOW]", "MEDIUM": "[MED]", "HIGH": "[HIGH]"}

    icon     = decision_map.get(report["final_decision"], "[?]")
    risk_str = risk_map.get(report["risk_level"], report["risk_level"])

    print("\n" + "="*60)
    print(f"  FYRP -- FINAL DECISION REPORT")
    print("="*60)
    print(f"  Image      : {Path(report['image_path']).name}")
    print(f"  CNN Pred   : {report['cnn_prediction']}")
    print(f"  Decision   : {icon}  {report['final_decision']}")
    print(f"  Risk Level : {risk_str}  {report['risk_level']}")
    print(f"  Confidence : {report['confidence']}")
    override_str = ('YES -- ' + report['override_reason']) if report['override'] else 'No'
    print(f"  Override   : {override_str}")
    print(f"  Heatmap    : {report['heatmap_path'] or 'Not saved'}")
    print(f"  Elapsed    : {report['elapsed_seconds']}s")
    print("-"*60)
    reasoning = report['reasoning']
    short = reasoning[:300] + ('...' if len(reasoning) > 300 else '')
    print(f"  Reasoning  : {short}")
    print("="*60 + "\n")


# ────────────────────────────────────────────────────────────────────
#  CLI entry point
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="FYRP — Adversarial Face Detection Pipeline"
    )
    parser.add_argument("--image",      required=True,  help="Path to input face image")
    parser.add_argument("--checkpoint", default=None,   help="Path to model checkpoint .pth")
    parser.add_argument("--api_key",    default=None,   help="OpenAI API key (or set OPENAI_API_KEY)")
    parser.add_argument("--model",      default=None,    help="LLM model name (default: from LLM_MODEL env / provider default)")
    parser.add_argument("--no_heatmap", action="store_true", help="Skip Grad-CAM heatmap")
    parser.add_argument("--output_dir", default=None,   help="Output directory for results")

    args = parser.parse_args()

    result = run_pipeline(
        image_path=args.image,
        checkpoint=args.checkpoint,
        openai_api_key=args.api_key,
        model_name=args.model,
        save_heatmap=not args.no_heatmap,
        output_dir=args.output_dir,
    )
