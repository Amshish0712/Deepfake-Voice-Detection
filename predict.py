import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog

print(" Loading model...")
model = load_model("deepfake_voice_model_cnn.keras")
print(" Model loaded")

def extract_features(file_path, max_pad_len=174):
    try:
        print("Loading audio...")
        audio, sr = librosa.load(file_path, sr=16000, duration=5)
        print("Audio loaded")

        if len(audio) == 0:
            return None
        
        print(" Extracting MFCC...")
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        print("MFCC done")

        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        
        return mfcc
    
    except Exception as e:
        print("Error:", e)
        return None

root = Tk()
root.withdraw()
root.attributes('-topmost', True)

file_path = filedialog.askopenfilename(title="Select Audio File")

if not file_path:
    print("No file selected")
    exit()

print("\nSelected file:", file_path)

choice = input("\nDo you want to play audio? (y/n): ").lower()

if choice == 'y':
    print("Opening audio in media player...")
    os.startfile(file_path)   


features = extract_features(file_path)

if features is not None:
    features = features[np.newaxis, ..., np.newaxis]

    print("\nPredicting...")
    prediction = model.predict(features)[0][0]

    label = "Fake Voice" if prediction > 0.5 else "Real Voice "
    confidence = prediction if prediction > 0.5 else 1 - prediction

    print("\nPrediction:", label)

else:
    print("Feature extraction failed")