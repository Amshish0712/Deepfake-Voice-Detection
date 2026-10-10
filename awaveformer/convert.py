import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
from pathlib import Path
import os
import threading

def get_desktop_path():
    return Path.home() / "Desktop"

def convert_m4a_to_wav(input_path: str, output_path: str) -> str:
    command = [
        'ffmpeg',
        '-i', str(input_path),
        '-acodec', 'pcm_s16le',
        '-ar', '44100',
        '-ac', '2',
        '-y',
        output_path
    ]
    
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed:\n{result.stderr}")
    
    return output_path

def select_and_convert():
    root = tk.Tk()
    root.withdraw()
    
    input_path = filedialog.askopenfilename(
        title="Select M4A File",
        filetypes=[("M4A files", "*.m4a"), ("All files", "*.*")]
    )
    
    if not input_path:
        print("No file selected.")
        return
    
    input_file = Path(input_path)
    desktop = get_desktop_path()
    output_path = str(desktop / input_file.with_suffix('.wav').name)
    
    print(f"Converting: {input_path}")
    print(f"Saving to: {output_path}")
    
    try:
        convert_m4a_to_wav(input_path, output_path)
        messagebox.showinfo(
            "Success",
            f"Conversion complete!\nSaved to:\n{output_path}"
        )
        print(f"Done! File saved to: {output_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Conversion failed:\n{str(e)}")
        print(f"Error: {e}")

if __name__ == "__main__":
    select_and_convert()