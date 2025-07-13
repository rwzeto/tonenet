#!/usr/bin/env python3
"""
main.py — Mandarin tone detector (ToneNet weights, Keras 3-safe)

• Enter → start / Enter → stop recording
• Builds a 225×225×3 mel-spectrogram image exactly like the paper
• Re-creates the ToneNet CNN, loads weights from ToneNet.h5
• Predicts Tone 1-4 and saves a spectrogram-plus-pitch PNG
"""

from __future__ import annotations
import os, io, time, warnings, pathlib
import numpy as np
import sounddevice as sd
import librosa, librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Flatten, Dense

# ───────── constants ───────── #
FS = 16_000
IMG_SIZE = 225  # ToneNet uses 225×225 RGB “mel images”
MODEL_PATH = "ToneNet.h5"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ───────── ToneNet architecture (clean, Keras 3-compatible) ───────── #
def build_tonet() -> tf.keras.Model:
    model = Sequential(
        name="ToneNet",
        layers=[
            Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            Conv2D(64, (5, 5), strides=3, padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPool2D((3, 3), strides=3, padding="same"),
            Conv2D(128, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPool2D((2, 2), strides=2, padding="same"),
            Conv2D(256, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPool2D((2, 2), strides=2, padding="same"),
            Conv2D(256, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPool2D((2, 2), strides=2, padding="same"),
            Conv2D(512, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPool2D((2, 2), strides=2, padding="same"),
            Flatten(),
            Dense(1024),
            BatchNormalization(),
            Activation("relu"),
            Dense(1024),
            BatchNormalization(),
            Activation("relu"),
            Dense(4, activation="softmax"),
        ],
    )
    return model


print("▶ Building ToneNet skeleton …")
model = build_tonet()

print("▶ Loading weights from", MODEL_PATH)
model.load_weights(MODEL_PATH)


# ───────── recording (Enter → Enter) ───────── #
def record(fs=FS) -> np.ndarray:
    input("Press ⏎ to start recording …")
    print("Recording — press ⏎ again to stop.")
    buf: list[np.ndarray] = []
    with sd.InputStream(samplerate=fs, channels=1, dtype="float32", callback=lambda d, *_: buf.append(d.copy())):
        input()
    audio = np.concatenate(buf).ravel()
    print(f"Captured {audio.size/fs:.2f}s")
    return librosa.util.normalize(audio)


# ───────── feature: mel image + simple pitch for overlay ───────── #
def mel_image(y: np.ndarray, sr=FS) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=2048, hop_length=16, fmin=50, fmax=350)
    S_db = librosa.power_to_db(S, ref=np.max)

    # draw to an RGB PIL image exactly 225×225
    fig = plt.figure(figsize=(2.25, 2.25), dpi=100)
    librosa.display.specshow(S_db, sr=sr, fmin=50, fmax=350)
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    img = Image.open(buf).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img).astype("float32")


# ───────── feature: quick pitch for overlay ───────── #
def crude_pitch(y: np.ndarray, sr: int = FS):
    """Cheap autocorrelation-based F0 track (overlay only)."""
    from scipy.signal import correlate
    from scipy.interpolate import interp1d

    FMIN, FMAX = 80, 400
    win, hop = int(0.03 * sr), int(0.01 * sr)

    # librosa ≥0.10: keyword args are required
    frames = librosa.util.frame(y, frame_length=win, hop_length=hop).T

    f0 = []
    for frame in frames:
        frame = frame - frame.mean()
        if np.allclose(frame, 0):
            f0.append(np.nan)
            continue
        corr = correlate(frame, frame, mode="full")[win - 1 :]
        corr /= corr[0] + 1e-9
        lag_min, lag_max = int(sr / FMAX), int(sr / FMIN)
        corr[:lag_min] = 0
        peak = np.argmax(corr[lag_min:lag_max]) + lag_min
        f0.append(sr / peak if corr[peak] > 0.3 else np.nan)

    t = np.arange(len(f0)) * hop / sr
    voiced = ~np.isnan(f0)
    if voiced.sum() < 3:
        return t, np.array([])

    f0_interp = interp1d(t[voiced], np.array(f0)[voiced], fill_value="extrapolate")
    return t, f0_interp(t)


# ───────── one-shot inference ───────── #
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    y = record()
    img = mel_image(y)  # (225,225,3)
    logits = model.predict(img[None] / 255.0, verbose=0)[0]
    tone = int(np.argmax(logits) + 1)

    # overlay + save
    t, f0 = crude_pitch(y)
    out = os.path.join(OUT_DIR, f"spectrogram-{time.strftime('%Y%m%d-%H%M%S')}.png")
    plt.figure(figsize=(6, 3))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predicted tone: {tone}")
    if f0.size:
        plt.plot(t * IMG_SIZE / y.size * FS, IMG_SIZE - f0 / 350 * IMG_SIZE, "w", lw=2)  # rough x-mapping  # rough y-mapping
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Detected tone → {tone}\nFigure saved  → {out}")


if __name__ == "__main__":
    main()
