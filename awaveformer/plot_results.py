import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay
from model import AWaveFormerLite
from dataset import AudioDeepfakeDataset
from torch.utils.data import DataLoader
import os

os.makedirs('figures', exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT   = 'checkpoints/ep05_eer3.79.pt'

epochs     = [1,      2,      3,      4,      5     ]
train_loss = [0.1535, 0.0775, 0.0636, 0.0539, 0.0486]
eer        = [5.24,   4.25,   4.83,   3.83,   3.79  ]
acc        = [94.8,   95.6,   95.0,   96.1,   94.2  ]

steps = list(range(500, 27001, 500))
loss_e1 = [0.4715,0.4007,0.3553,0.3413,0.3172,0.3073,0.3035,0.2810,0.2790,
           0.2741,0.2601,0.2575,0.2482,0.2440,0.2392,0.2348,0.2310,0.2267,
           0.2218,0.2150,0.2105,0.2070,0.2059,0.2024,0.2014,0.1993,0.1951,
           0.1935,0.1923,0.1894,0.1863,0.1845,0.1824,0.1819,0.1810,0.1788,
           0.1769,0.1757,0.1752,0.1725,0.1706,0.1688,0.1670,0.1653,0.1643,
           0.1635,0.1622,0.1610,0.1595,0.1579,0.1566,0.1557,0.1547]
loss_e2 = [0.1067,0.1053,0.1016,0.0995,0.0924,0.0911,0.1014,0.0961,0.0946,
           0.0932,0.0936,0.0912,0.0911,0.0887,0.0902,0.0901,0.0902,0.0886,
           0.0896,0.0879,0.0862,0.0856,0.0859,0.0850,0.0841,0.0830,0.0831,
           0.0832,0.0825,0.0835,0.0825,0.0823,0.0824,0.0816,0.0827,0.0822,
           0.0817,0.0818,0.0817,0.0809,0.0805,0.0804,0.0797,0.0799,0.0806,
           0.0803,0.0799,0.0798,0.0797,0.0792,0.0787,0.0787,0.0779]
loss_e3 = [0.0840,0.0981,0.0812,0.0805,0.0804,0.0758,0.0755,0.0726,0.0721,
           0.0694,0.0685,0.0708,0.0699,0.0692,0.0676,0.0689,0.0696,0.0698,
           0.0694,0.0693,0.0681,0.0673,0.0683,0.0691,0.0698,0.0696,0.0688,
           0.0681,0.0670,0.0665,0.0673,0.0666,0.0653,0.0647,0.0655,0.0653,
           0.0660,0.0651,0.0647,0.0644,0.0643,0.0643,0.0638,0.0656,0.0649,
           0.0644,0.0635,0.0639,0.0631,0.0632,0.0628,0.0627,0.0636]
loss_e4 = [0.0538,0.0447,0.0530,0.0450,0.0463,0.0511,0.0528,0.0565,0.0556,
           0.0554,0.0576,0.0612,0.0599,0.0575,0.0580,0.0576,0.0580,0.0583,
           0.0594,0.0597,0.0593,0.0584,0.0586,0.0575,0.0580,0.0581,0.0577,
           0.0571,0.0558,0.0562,0.0561,0.0556,0.0552,0.0555,0.0554,0.0558,
           0.0552,0.0547,0.0558,0.0554,0.0552,0.0544,0.0543,0.0548,0.0549,
           0.0547,0.0548,0.0546,0.0550,0.0552,0.0549,0.0548,0.0542]
loss_e5 = [0.0849,0.0684,0.0741,0.0760,0.0682,0.0645,0.0608,0.0615,0.0573,
           0.0549,0.0527,0.0517,0.0512,0.0512,0.0508,0.0512,0.0522,0.0513,
           0.0525,0.0518,0.0525,0.0532,0.0521,0.0512,0.0514,0.0521,0.0523,
           0.0528,0.0524,0.0519,0.0509,0.0508,0.0502,0.0502,0.0506,0.0518,
           0.0516,0.0507,0.0503,0.0503,0.0496,0.0492,0.0493,0.0488,0.0482,
           0.0484,0.0487,0.0486,0.0490,0.0483,0.0489,0.0490,0.0490]

# ── Figure 1: step-level loss ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
for i, (loss, col) in enumerate(zip(
        [loss_e1,loss_e2,loss_e3,loss_e4,loss_e5], colors), 1):
    ax.plot(steps[:len(loss)], loss, linewidth=1.2,
            color=col, label=f'Epoch {i}')
ax.set_xlabel('Step')
ax.set_ylabel('Cross-Entropy Loss')
ax.set_title('Training Loss per Step')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/loss_per_step.png', dpi=150, bbox_inches='tight')
print("Saved: figures/loss_per_step.png")
plt.close()

# ── Figure 2: epoch summary ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('AWaveFormer Lite — Training Summary', fontsize=13, fontweight='bold')

axes[0].plot(epochs, train_loss, 'o-', color='steelblue', linewidth=2, markersize=7)
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_xticks(epochs)
axes[0].grid(True, alpha=0.3)
for x, y in zip(epochs, train_loss):
    axes[0].annotate(f'{y:.4f}', (x, y), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)

axes[1].plot(epochs, eer, 'o-', color='tomato', linewidth=2, markersize=7)
axes[1].set_title('EER (%)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('EER (%)')
axes[1].set_xticks(epochs)
axes[1].grid(True, alpha=0.3)
for x, y in zip(epochs, eer):
    axes[1].annotate(f'{y:.2f}%', (x, y), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)

axes[2].plot(epochs, acc, 'o-', color='seagreen', linewidth=2, markersize=7)
axes[2].set_title('Accuracy (%)')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Accuracy (%)')
axes[2].set_xticks(epochs)
axes[2].set_ylim([90, 100])
axes[2].grid(True, alpha=0.3)
for x, y in zip(epochs, acc):
    axes[2].annotate(f'{y:.1f}%', (x, y), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('figures/epoch_summary.png', dpi=150, bbox_inches='tight')
print("Saved: figures/epoch_summary.png")
plt.close()

# ── Load model ────────────────────────────────────────────────────────────────
print("\nLoading model...")
model = AWaveFormerLite(num_mwpm=6).to(DEVICE)
ckpt  = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()

ds     = AudioDeepfakeDataset('/home/system12/dataset', 'validation')
loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)

all_scores, all_labels = [], []
with torch.no_grad():
    for wav, label in loader:
        probs = torch.softmax(model(wav.to(DEVICE)), dim=1)[:, 1]
        all_scores.extend(probs.cpu().numpy())
        all_labels.extend(label.numpy())

scores = np.array(all_scores)
labels = np.array(all_labels)
preds  = (scores >= 0.5).astype(int)

# ── Figure 3: confusion matrix ────────────────────────────────────────────────
cm  = confusion_matrix(labels, preds)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Fake', 'Real'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix')
total = cm.sum()
ax.set_xlabel(f'Predicted Label\n\nTotal: {total} | '
              f'Correct: {cm[0,0]+cm[1,1]} | '
              f'Accuracy: {(cm[0,0]+cm[1,1])/total*100:.1f}%')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Saved: figures/confusion_matrix.png")
plt.close()

# ── Figure 4: ROC curve ───────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
fnr = 1.0 - tpr
eer_idx = np.nanargmin(np.abs(fpr - fnr))
eer_val = (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color='steelblue', linewidth=2, label='ROC curve')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
ax.scatter(fpr[eer_idx], tpr[eer_idx], color='tomato', zorder=5,
           s=100, label=f'EER = {eer_val:.2f}%')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/roc_curve.png', dpi=150, bbox_inches='tight')
print("Saved: figures/roc_curve.png")
plt.close()

# ── Figure 5: score distribution ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(scores[labels==1], bins=60, alpha=0.6, color='steelblue',
        label='Real', density=True)
ax.hist(scores[labels==0], bins=60, alpha=0.6, color='tomato',
        label='Fake', density=True)
ax.axvline(0.5, color='black', linestyle='--', linewidth=1.5,
           label='Threshold = 0.5')
ax.set_xlabel('P(Real) score')
ax.set_ylabel('Density')
ax.set_title('Score Distribution — Real vs Fake (Validation Set)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/score_distribution.png', dpi=150, bbox_inches='tight')
print("Saved: figures/score_distribution.png")
plt.close()

print("\n" + "="*45)
print("  FINAL RESULTS — Validation Set")
print("="*45)
print("  EER      :", round(eer_val, 2), "percent")
print("  Accuracy :", round((preds==labels).mean()*100, 1), "percent")
print("  Real correct:", cm[1][1], "/", cm[1].sum())
print("  Fake correct:", cm[0][0], "/", cm[0].sum())
print("="*45)
print("\nAll figures saved in ./figures/")
