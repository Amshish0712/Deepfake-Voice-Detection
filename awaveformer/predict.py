import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import os
import sys

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT   = os.path.join(os.path.dirname(__file__), 'checkpoints', 'ep05_eer3.79.pt')

def load_wav(path):
    import torchaudio
    TARGET_SR = 16000
    MAX_LEN   = 67000
    wav, sr = torchaudio.load(path)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    wav = wav.mean(0)
    if wav.shape[0] > MAX_LEN:
        wav = wav[:MAX_LEN]
    else:
        wav = torch.nn.functional.pad(wav, (0, MAX_LEN - wav.shape[0]))
    return wav

def load_model():
    sys.path.insert(0, os.path.dirname(__file__))
    from model import AWaveFormerLite
    m = AWaveFormerLite(num_mwpm=6).to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE)
    state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    m.load_state_dict(state)
    m.eval()
    return m

# ── Build GUI first, then load model ─────────────────────────────────────────
root = tk.Tk()
root.title("AWaveFormer — Audio Deepfake Detector")
root.geometry("540x420")
root.resizable(False, False)
root.configure(bg="#f4f4f4")

# header
tk.Label(root, text="Audio Deepfake Detector",
         font=("Segoe UI", 20, "bold"),
         bg="#f4f4f4", fg="#1a1a1a").pack(pady=(28, 2))

tk.Label(root, text="Based on AWaveFormer  |  IEEE TASLP 2025",
         font=("Segoe UI", 9),
         bg="#f4f4f4", fg="#888888").pack()

tk.Frame(root, height=1, bg="#cccccc").pack(fill=tk.X, padx=30, pady=14)

# file path display
file_var = tk.StringVar(value="No file selected")
tk.Label(root, textvariable=file_var,
         font=("Segoe UI", 9), bg="#f4f4f4",
         fg="#555555", wraplength=480).pack(pady=(0, 10))

# result display
result_label = tk.Label(root, text="—",
                        font=("Segoe UI", 48, "bold"),
                        bg="#f4f4f4", fg="#aaaaaa")
result_label.pack()

conf_label = tk.Label(root, text="",
                      font=("Segoe UI", 14),
                      bg="#f4f4f4", fg="#444444")
conf_label.pack(pady=(2, 0))

detail_label = tk.Label(root, text="",
                        font=("Segoe UI", 10),
                        bg="#f4f4f4", fg="#888888")
detail_label.pack(pady=(2, 0))

# buttons
btn_frame = tk.Frame(root, bg="#f4f4f4")
btn_frame.pack(pady=22)

def predict(path):
    try:
        result_label.config(text="...", fg="#aaaaaa")
        conf_label.config(text="Analysing...")
        detail_label.config(text="")
        root.update()

        wav = load_wav(path).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(wav)
            probs  = torch.softmax(logits, dim=1)[0]

        p_real = probs[1].item()
        p_fake = probs[0].item()
        pred   = "REAL" if p_real > p_fake else "FAKE"
        conf   = max(p_real, p_fake) * 100

        if pred == "REAL":
            result_label.config(text="REAL", fg="#1a7a1a")
        else:
            result_label.config(text="FAKE", fg="#cc1a1a")

        conf_label.config(text=f"Confidence: {conf:.1f}%")
        detail_label.config(
            text=f"P(Real) = {p_real*100:.1f}%     P(Fake) = {p_fake*100:.1f}%")

    except Exception as e:
        messagebox.showerror("Error", f"Could not process file:\n{e}")
        result_label.config(text="—", fg="#aaaaaa")
        conf_label.config(text="")
        detail_label.config(text="")

def browse():
    path = filedialog.askopenfilename(
        title="Select an audio file",
        filetypes=[
            ("WAV files",   "*.wav"),
            ("FLAC files",  "*.flac"),
            ("MP3 files",   "*.mp3"),
            ("All files",   "*.*")
        ]
    )
    if path:
        file_var.set(os.path.basename(path))
        predict(path)

def clear():
    file_var.set("No file selected")
    result_label.config(text="—", fg="#aaaaaa")
    conf_label.config(text="")
    detail_label.config(text="")

tk.Button(btn_frame, text="  Select Audio File  ",
          command=browse,
          font=("Segoe UI", 12, "bold"),
          bg="#2255cc", fg="white",
          padx=16, pady=8,
          relief="flat", cursor="hand2",
          activebackground="#1a44aa",
          activeforeground="white").pack(side=tk.LEFT, padx=8)

tk.Button(btn_frame, text="  Clear  ",
          command=clear,
          font=("Segoe UI", 12),
          bg="#888888", fg="white",
          padx=16, pady=8,
          relief="flat", cursor="hand2",
          activebackground="#666666",
          activeforeground="white").pack(side=tk.LEFT, padx=8)

# footer
tk.Label(root,
         text=f"Model: ep05_eer3.79.pt  |  EER: 3.79%  |  Device: {DEVICE.upper()}",
         font=("Segoe UI", 8),
         bg="#f4f4f4", fg="#bbbbbb").pack(side=tk.BOTTOM, pady=8)

# ── Load model after GUI is ready ─────────────────────────────────────────────
status = tk.Label(root, text="Loading model...",
                  font=("Segoe UI", 9),
                  bg="#f4f4f4", fg="#e07000")
status.pack(side=tk.BOTTOM, pady=2)
root.update()

try:
    model = load_model()
    status.config(text="Model loaded and ready", fg="#1a7a1a")
except Exception as e:
    messagebox.showerror("Model Load Error",
                         f"Could not load model:\n{e}\n\n"
                         f"Make sure checkpoints/ep05_eer3.79.pt exists.")
    sys.exit(1)

root.mainloop()
