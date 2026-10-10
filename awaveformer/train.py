import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from model import AWaveFormerLite
from dataset import AudioDeepfakeDataset

DATA_DIR     = '/home/system12/dataset'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS       = 5
BATCH_SIZE   = 2
GRAD_ACCUM   = 4
LR           = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4
SAVE_DIR     = 'checkpoints'

def compute_eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2 * 100)

def evaluate(model, loader):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for wav, label in loader:
            wav   = wav.to(DEVICE)
            probs = torch.softmax(model(wav), dim=1)[:, 1]
            all_scores.extend(probs.cpu().numpy())
            all_labels.extend(label.numpy())
    scores = np.array(all_scores)
    labels = np.array(all_labels)
    eer    = compute_eer(scores, labels)
    acc    = ((scores >= 0.5).astype(int) == labels).mean() * 100
    return eer, acc

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Device: {DEVICE}\n")

    train_ds = AudioDeepfakeDataset(DATA_DIR, 'training')
    val_ds   = AudioDeepfakeDataset(DATA_DIR, 'validation')
    test_ds  = AudioDeepfakeDataset(DATA_DIR, 'testing')

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    model = AWaveFormerLite(num_mwpm=6).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,} params\n")

    opt       = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    best_eer  = float('inf')
    best_ckpt = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        running_loss = 0.0

        for step, (wav, label) in enumerate(train_dl, 1):
            wav, label = wav.to(DEVICE), label.to(DEVICE)

            # no autocast — pascal GPU has no tensor cores, fp32 throughout
            loss = criterion(model(wav), label) / GRAD_ACCUM
            loss.backward()
            running_loss += loss.item() * GRAD_ACCUM

            if step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

            if step % 500 == 0:
                print(f"  Epoch {epoch} | step {step}/{len(train_dl)} "
                      f"| loss {running_loss/step:.4f}")

        val_eer, val_acc = evaluate(model, val_dl)
        avg_loss = running_loss / len(train_dl)
        print(f"\nEpoch {epoch:02d} | loss={avg_loss:.4f} "
              f"| val_EER={val_eer:.2f}% | val_acc={val_acc:.1f}%")

        ckpt = os.path.join(SAVE_DIR, f'ep{epoch:02d}_eer{val_eer:.2f}.pt')
        torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                    'val_eer': val_eer}, ckpt)

        if val_eer < best_eer:
            best_eer  = val_eer
            best_ckpt = ckpt
            print(f"  *** Best so far: {val_eer:.2f}% -> {ckpt}")
        print()

    print(f"Loading best checkpoint: {best_ckpt}")
    model.load_state_dict(torch.load(best_ckpt)['model_state'])
    test_eer, test_acc = evaluate(model, test_dl)
    print(f"\n{'='*45}")
    print(f"  TEST EER      : {test_eer:.2f}%")
    print(f"  TEST Accuracy : {test_acc:.1f}%")
    print(f"{'='*45}")

if __name__ == '__main__':
    main()
