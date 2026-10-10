import torch
from model import AWaveFormerLite
from dataset import AudioDeepfakeDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix

DEVICE = 'cuda'
CKPT   = 'checkpoints/ep05_eer3.79.pt'

def compute_eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2 * 100)

ds     = AudioDeepfakeDataset('/home/system12/dataset', 'validation')
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4)

model  = AWaveFormerLite(num_mwpm=6).to(DEVICE)
ckpt   = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()

all_scores, all_labels = [], []
with torch.no_grad():
    for wav, label in loader:
        probs = torch.softmax(model(wav.to(DEVICE)), dim=1)[:, 1]
        all_scores.extend(probs.cpu().numpy())
        all_labels.extend(label.numpy())

scores = np.array(all_scores)
labels = np.array(all_labels)
preds  = (scores >= 0.5).astype(int)

eer = compute_eer(scores, labels)
acc = (preds == labels).mean() * 100
cm  = confusion_matrix(labels, preds)

print("Test EER      :", round(eer, 2), "percent")
print("Test Accuracy :", round(acc, 1), "percent")
print("Confusion Matrix:")
print("  Real correct:", cm[1][1], "/", cm[1].sum())
print("  Fake correct:", cm[0][0], "/", cm[0].sum())
