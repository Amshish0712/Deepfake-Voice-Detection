import os
import torch
import torchaudio
from torch.utils.data import Dataset

TARGET_SR = 16000
MAX_LEN   = 67000

def load_wav(path):
    wav, sr = torchaudio.load(path)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    wav = wav.mean(0)
    if wav.shape[0] > MAX_LEN:
        wav = wav[:MAX_LEN]
    else:
        wav = torch.nn.functional.pad(wav, (0, MAX_LEN - wav.shape[0]))
    return wav

class AudioDeepfakeDataset(Dataset):
    def __init__(self, root, split):
        self.samples = []
        for label_name, label in [('real', 1), ('fake', 0)]:
            folder = os.path.join(root, split, label_name)
            for fname in os.listdir(folder):
                if fname.lower().endswith('.wav'):
                    self.samples.append((os.path.join(folder, fname), label))
        real = sum(1 for _, l in self.samples if l == 1)
        fake = sum(1 for _, l in self.samples if l == 0)
        print(f"[{split}] {len(self.samples)} files  (real={real}, fake={fake})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            return load_wav(path), label
        except Exception as e:
            print(f"WARNING: could not load {path}: {e}")
            return torch.zeros(MAX_LEN), label
