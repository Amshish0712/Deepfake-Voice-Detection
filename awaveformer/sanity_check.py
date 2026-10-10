import torch, time
from dataset import AudioDeepfakeDataset
from model import AWaveFormerLite

DATA_DIR = '/home/system12/dataset'
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*50)
print(f"Device : {DEVICE}")
if DEVICE == 'cuda':
    p = torch.cuda.get_device_properties(0)
    print(f"GPU    : {p.name}  ({p.total_memory/1024**3:.1f} GB)")

print("\n--- Dataset ---")
for split in ('training', 'validation', 'testing'):
    ds = AudioDeepfakeDataset(DATA_DIR, split)
    wav, label = ds[0]
    print(f"  {split}: wav={wav.shape}, label={label} OK")

print("\n--- Forward pass (batch=2) ---")
model = AWaveFormerLite(num_mwpm=6).to(DEVICE)
dummy = torch.randn(2, 67000).to(DEVICE)
t0    = time.time()
with torch.no_grad():
    out = model(dummy)
print(f"  Output : {out.shape}  ({(time.time()-t0)*1000:.0f} ms)")
print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

if DEVICE == 'cuda':
    print(f"  VRAM used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

print("\n  All OK — run: python train.py")
print("="*50)
