"""
main.py — Mandarin tone detector (ToneNet weights, Keras-3 safe)

Workflow
────────
1. ⏎  → start ⏎  → stop recording (16 kHz mono).
2. Build a 225×225 viridis-coloured mel-spectrogram image
   with the exact Figure settings used in ToneNet’s training code.
3. Apply “caffe” preprocessing (RGB→BGR, ImageNet mean subtraction).
4. Run the pre-trained ToneNet CNN and print the raw probabilities
   plus the predicted Tone 1-4.
"""

from __future__ import annotations
import io, warnings
import numpy as np
import sounddevice as sd
import librosa, librosa.display
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Flatten, Dense

# ───────────────────────── constants ───────────────────────── #
FS = 16_000  # input sample-rate
IMG_SIZE = 225  # ToneNet expects 225×225
MODEL_PATH = "ToneNet.h5"  # your downloaded weights


# ─────────────────── ToneNet network skeleton ────────────────── #
def build_tonet() -> tf.keras.Model:
    return Sequential(
        [
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
        name="ToneNet",
    )


print("▶ Building ToneNet …")
model = build_tonet()
print(f"▶ Loading weights from {MODEL_PATH}")
model.load_weights(MODEL_PATH)


# ─────────────────────── audio capture ─────────────────────── #
def record(fs: int = FS) -> np.ndarray:
    input("Press ⏎ to start recording … ")
    print("Recording — press ⏎ again to stop.")
    buf: list[np.ndarray] = []
    with sd.InputStream(
        samplerate=fs,
        channels=1,
        dtype="float32",
        callback=lambda d, *_: buf.append(d.copy()),
    ):
        input()
    audio = np.concatenate(buf).ravel()
    print(f"Captured {audio.size / fs:.2f} s")
    return librosa.util.normalize(audio)


# ─────────────── build 225×225 mel-spectrogram image ─────────────── #
def mel_image(y: np.ndarray, sr: int = FS) -> np.ndarray:
    """
    Returns a (225, 225, 3) uint8 RGB image produced exactly like
    `Feature_Extraction.py` in the original repo.
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=64,
        n_fft=2048,
        hop_length=16,
        fmin=50,
        fmax=350,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(2.25, 2.25), dpi=100)  # 225 px
    librosa.display.specshow(S_db, sr=sr, fmin=50, fmax=350)  # default 'viridis'
    # Remove ticks & margins exactly like the training script
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)

    img = Image.open(buf).convert("RGB")
    return np.asarray(img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8)


# ─────────── ToneNet “caffe” preprocessing helper ─────────── #
def preprocess_caffe(rgb: np.ndarray) -> np.ndarray:
    """
    1. RGB → BGR
    2. subtract ImageNet mean [103.939, 116.779, 123.68]
    3. keep pixel range 0-255 (no /255 scaling)
    """
    bgr = rgb[..., ::-1].astype("float32")
    mean = np.array([103.939, 116.779, 123.68], dtype="float32")
    return bgr - mean


def detect_tone(
    audio: np.ndarray,
    sr: int = FS,
    return_probs: bool = False,
) -> int | tuple[int, np.ndarray]:
    """
    Detects the Mandarin tone (1-4) for a given audio segment.

    Parameters
    ----------
    audio : np.ndarray
        1-D mono waveform (float32, any length, range -1…1).
        If the audio comes directly from sounddevice/librosa it’s
        already float32; otherwise call `librosa.util.normalize` first.
    sr : int, default FS (16 000)
        Sample rate of the `audio` array.
    return_probs : bool, default False
        If True, also return the raw soft-max probabilities.

    Returns
    -------
    int
        Detected tone number (1–4).
        If `return_probs` is True, returns
        `(tone: int, probs: np.ndarray)` instead.
    """
    # Build ToneNet input exactly like training
    rgb_img = mel_image(audio, sr=sr)  # (225, 225, 3) uint8 RGB
    net_in = preprocess_caffe(rgb_img)  # float32 BGR, mean-centered

    # Forward pass (batch dimension added)
    probs = model.predict(net_in[None], verbose=0)[0]
    tone = int(np.argmax(probs) + 1)

    if return_probs:
        return tone, probs
    return tone


# ─────────────────────────── main ─────────────────────────── #
def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    audio = record()
    rgb_img = mel_image(audio)  # uint8 RGB
    net_in = preprocess_caffe(rgb_img)  # float32 BGR, mean-centered

    probs = model.predict(net_in[None], verbose=0)[0]
    print("Raw probabilities:", np.round(probs, 3))
    tone = int(np.argmax(probs) + 1)
    print(f"Detected tone → {tone}")


if __name__ == "__main__":
    main()
