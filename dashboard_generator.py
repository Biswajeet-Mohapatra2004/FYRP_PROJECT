import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def generate_dynamic_dashboard(image_path, gradcam_path, report, output_path):
    # Load original image
    orig_img = cv2.imread(image_path)
    if orig_img is not None:
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        orig_img = cv2.resize(orig_img, (224, 224))
    else:
        orig_img = np.zeros((224, 224, 3), dtype=np.uint8)
        
    # Load gradcam image
    gradcam_img = None
    if gradcam_path and os.path.exists(gradcam_path):
        gradcam_img = cv2.imread(gradcam_path)
    
    if gradcam_img is not None:
        gradcam_img = cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB)
        gradcam_img = cv2.resize(gradcam_img, (224, 224))
    else:
        gradcam_img = np.zeros((224, 224, 3), dtype=np.uint8)
        
    # Extract values from report
    params = ['CNN Confidence (Inverted)', 'Anomaly Score', 'Embedding Drift', 'Image Blur/Quality']
    td = report.get('threshold_details', {})
    factors = td.get('threshold_components', {})
    
    v_conf = factors.get('confidence_factor', 0.0)
    v_anom = factors.get('anomaly_factor', 0.0)
    v_drift = factors.get('drift_factor', 0.0)
    v_qual = factors.get('quality_factor', 0.0)
    values = [v_conf, v_anom, v_drift, v_qual]
    weights = [0.35, 0.30, 0.20, 0.15]
    
    T = td.get('computed_threshold', 0.0)
    final_decision = report.get('final_decision', 'UNKNOWN')
    risk_level = report.get('risk_level', 'UNKNOWN')
    
    # Set up figure
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(f'Unified XAI Dashboard: Explaining the "{final_decision}" Verdict', fontsize=18, fontweight='bold', y=1.05)
    
    # Plot 1: Original Image
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(orig_img)
    ax1.set_title('1. Input Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Plot 2: Grad-CAM
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(gradcam_img)
    ax2.set_title('2. Spatial XAI (Grad-CAM)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Explanation text based on risk
    explanation = "Heatmap shows CNN focus."
    if final_decision in ["ADVERSARIAL", "SUSPICIOUS"]:
        explanation += "\nNotice irregular focus areas\nindicating potential adversarial\nperturbation."
    else:
        explanation += "\nFocus is primarily on\nlegitimate facial features."
        
    ax2.text(0.5, -0.15, explanation, transform=ax2.transAxes, ha='center', va='top', 
             fontsize=11, bbox=dict(facecolor='#3498db' if risk_level == 'LOW' else '#e74c3c', alpha=0.2, boxstyle='round,pad=0.5'))
    
    # Plot 3: Parameters
    ax3 = plt.subplot(1, 3, 3)
    y_pos = np.arange(len(params))
    bars = ax3.barh(y_pos, values, color=['#e74c3c', '#e67e22', '#f1c40f', '#9b59b6'])
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(params, fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    ax3.set_xlim(0, 1.1)
    ax3.set_title('3. Mathematical XAI (Threshold Engine)', fontsize=14, fontweight='bold')
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax3.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                 f'Score: {width:.2f} (Wt: {int(weights[i]*100)}%)',
                 ha='left', va='center', fontsize=11, fontweight='bold')
                 
    # Verdict box
    color_box = '#e74c3c' if risk_level in ['HIGH', 'MEDIUM'] else '#2ecc71'
    ax3.text(0.5, -0.15, f"Computed Threshold (T) = {T:.2f}\nRisk Level: {risk_level}", 
             transform=ax3.transAxes, ha='center', va='top', fontsize=12, fontweight='bold',
             bbox=dict(facecolor=color_box, alpha=0.3, boxstyle='round,pad=0.5'))
             
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path
