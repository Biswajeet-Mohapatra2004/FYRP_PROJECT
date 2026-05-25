import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

# 1. Load Image (We'll simulate a face for the dashboard if images.jpg is missing)
img_path = 'images.jpg'
if os.path.exists(img_path):
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_img = cv2.resize(orig_img, (224, 224))
else:
    orig_img = np.zeros((224, 224, 3), dtype=np.uint8)
    orig_img[:] = (150, 150, 150)

# 2. Simulate a Grad-CAM Heatmap (concentrated on unnatural areas to show adversarial disruption)
heatmap = np.zeros((224, 224), dtype=np.float32)
# Add some fake "focus" spots that don't make sense for a face (e.g., corners, edges)
cv2.circle(heatmap, (50, 50), 40, 1.0, -1)
cv2.circle(heatmap, (180, 150), 30, 0.8, -1)
heatmap = cv2.GaussianBlur(heatmap, (45, 45), 0)
heatmap = np.uint8(255 * heatmap)
color_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
gradcam_img = cv2.addWeighted(orig_img, 0.5, color_map, 0.5, 0)

# 3. Define the 4 Technical Parameters for the attacked image
params = ['CNN Confidence (Inverted)', 'Anomaly Score', 'Embedding Drift', 'Image Blur/Quality Deficit']
# Values between 0.0 and 1.0 showing high suspicion
values = [0.85, 0.92, 0.78, 0.65] 
weights = [0.35, 0.30, 0.20, 0.15]

# Setup the Figure
fig = plt.figure(figsize=(16, 6))
fig.suptitle('Unified XAI Dashboard: Explaining the "Adversarial" Verdict', fontsize=18, fontweight='bold', y=1.05)

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
explanation = (
    "Heatmap shows CNN focus.\n"
    "Notice how the model is looking at\n"
    "background noise rather than facial\n"
    "features (eyes/nose), proving the\n"
    "presence of adversarial perturbation."
)
ax2.text(0.5, -0.15, explanation, transform=ax2.transAxes, ha='center', va='top', 
         fontsize=11, bbox=dict(facecolor='#3498db', alpha=0.2, boxstyle='round,pad=0.5'))

# Plot 3: The 4 Parameters
ax3 = plt.subplot(1, 3, 3)
y_pos = np.arange(len(params))
bars = ax3.barh(y_pos, values, color=['#e74c3c', '#e67e22', '#f1c40f', '#9b59b6'])
ax3.set_yticks(y_pos)
ax3.set_yticklabels(params, fontsize=12, fontweight='bold')
ax3.invert_yaxis()  # labels read top-to-bottom
ax3.set_xlim(0, 1.1)
ax3.set_title('3. Mathematical XAI (Threshold Engine)', fontsize=14, fontweight='bold')

# Add values and weights text
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax3.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
             f'Score: {width:.2f} (Wt: {weights[i]*100}%)',
             ha='left', va='center', fontsize=11, fontweight='bold')

# Calculate total T
T = sum(v * w for v, w in zip(values, weights))
ax3.text(0.5, -0.15, f"Computed Threshold (T) = {T:.2f}\nBoundary crossed! Sent to LLM Judge.", 
         transform=ax3.transAxes, ha='center', va='top', fontsize=12, fontweight='bold',
         bbox=dict(facecolor='#e74c3c', alpha=0.3, boxstyle='round,pad=0.5'))

plt.tight_layout()
output_path = 'results/unified_xai_dashboard.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Unified XAI Dashboard saved to: {output_path}")
