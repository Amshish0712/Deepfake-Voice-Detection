import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from transformers import Wav2Vec2Model

class WaveletCoeffProcess(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwt     = DWT1DForward(wave='db3', J=1, mode='periodization')
        self.idwt    = DWT1DInverse(wave='db3', mode='periodization')
        self.conv_lo = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.conv_hi = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm    = nn.LayerNorm(dim)

    def forward(self, x):
        T       = x.shape[1]
        # cast to float32 — pytorch_wavelets backward doesn't support fp16
        x_f     = x.float()
        x_t     = x_f.transpose(1, 2)             # (B, D, T)
        lo, (hi,) = self.dwt(x_t)
        lo      = self.conv_lo(lo.float())
        hi      = self.conv_hi(hi.float())
        out     = self.idwt((lo, [hi]))            # (B, D, T')
        out     = out[:, :, :T].transpose(1, 2)   # (B, T, D)
        out     = out.to(x.dtype)                 # cast back to original dtype
        return self.norm(out + x)

class MultiScalePooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool3 = nn.MaxPool1d(3, stride=1, padding=1)
        self.pool5 = nn.MaxPool1d(5, stride=1, padding=2)
        self.pool7 = nn.MaxPool1d(7, stride=1, padding=3)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        out = (self.pool3(x_t) + self.pool5(x_t) + self.pool7(x_t)) / 3.0
        return out.transpose(1, 2)

class GatedFFN(nn.Module):
    def __init__(self, dim, expansion=2):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim * expansion, 1)
        self.conv2 = nn.Conv1d(dim * expansion, dim, 1)
        self.norm  = nn.LayerNorm(dim)

    def forward(self, x):
        x_t  = x.transpose(1, 2)
        h    = self.conv1(x_t)
        gate = torch.nn.functional.gelu(h) * h
        out  = self.conv2(gate).transpose(1, 2)
        return self.norm(out + x)

class MWPM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.wcp = WaveletCoeffProcess(dim)
        self.mpm = MultiScalePooling()
        self.gfn = GatedFFN(dim)

    def forward(self, x):
        return self.gfn(self.mpm(self.wcp(x)))

class AWaveFormerLite(nn.Module):
    def __init__(self, num_mwpm=6, d_model=768):
        super().__init__()
        self.w2v2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        for p in self.w2v2.parameters():
            p.requires_grad = False
        self.mwpms      = nn.ModuleList([MWPM(d_model) for _ in range(num_mwpm)])
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )

    def _pos_enc(self, T, D, device):
        pos = torch.arange(T, device=device).unsqueeze(1)
        i   = torch.arange(0, D, 2, device=device).unsqueeze(0)
        pe  = torch.zeros(1, T, D, device=device)
        pe[0, :, 0::2] = torch.sin(pos / 10000 ** (i / D))
        pe[0, :, 1::2] = torch.cos(pos / 10000 ** (i / D))
        return pe

    def forward(self, wav):
        with torch.no_grad():
            feats = self.w2v2(wav).last_hidden_state  # (B, T, 768)
        T, D  = feats.shape[1], feats.shape[2]
        feats = feats + self._pos_enc(T, D, feats.device)
        for mwpm in self.mwpms:
            feats = mwpm(feats)
        return self.classifier(feats.mean(dim=1))
    
