"""
tonenet.py — minimal helper exposing `detect_tone(audio, sr, return_probs=False)`

All CLI/recording utilities and the `__main__` entry-point have been removed,
leaving just what’s required to turn a 1-D waveform into a Tone 1-4 prediction.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import librosa
import librosa.display
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPool2D,
    Flatten,
    Dense,
)

# ───────────────────────── constants ───────────────────────── #
FS = 16_000  # expected sample-rate
IMG_SIZE = 225  # ToneNet input size
MODEL_PATH = Path("ToneNet.h5")  # pre-trained weights


# ─────────────────── ToneNet network skeleton ────────────────── #
def _build_tonet() -> tf.keras.Model:
    """Return the ToneNet architecture (weights loaded separately)."""
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


# Lazy-initialised global model (loaded on first use)
_model: tf.keras.Model | None = None


def _get_model() -> tf.keras.Model:
    global _model
    if _model is None:
        _model = _build_tonet()
        _model.load_weights(MODEL_PATH)
    return _model


# ─────────────── build 225×225 mel-spectrogram image ─────────────── #
def _mel_image(y: np.ndarray, sr: int = FS) -> np.ndarray:
    """
    Return a (225, 225, 3) uint8 RGB image produced exactly like the
    training pipeline (`Feature_Extraction.py`).
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

    import matplotlib.pyplot as plt  # local import keeps import time low

    fig = plt.figure(figsize=(2.25, 2.25), dpi=100)  # 225 px × 225 px
    librosa.display.specshow(S_db, sr=sr, fmin=50, fmax=350)

    # Remove ticks & margins exactly like the training script
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)

    img = Image.open(buf).convert("RGB")
    return np.asarray(img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8)


# ─────────── ToneNet “caffe” preprocessing helper ─────────── #
def _preprocess_caffe(rgb: np.ndarray) -> np.ndarray:
    """
    1. RGB → BGR
    2. subtract ImageNet mean [103.939, 116.779, 123.68]
    3. keep pixel range 0-255 (no /255 scaling)
    """
    bgr = rgb[..., ::-1].astype("float32")
    mean = np.array([103.939, 116.779, 123.68], dtype="float32")
    return bgr - mean


# ────────────────────────── public API ───────────────────────── #
def detect_tone(
    audio: np.ndarray,
    sr: int = FS,
    return_probs: bool = False,
) -> int | tuple[int, np.ndarray]:
    """
    Detect the Mandarin tone (1-4) of a 1-D mono waveform.

    Parameters
    ----------
    audio : np.ndarray
        Float32 waveform, any length. Values should already be in -1…1.
    sr : int, default 16 000
        Sample rate of `audio`. If not 16 kHz it will **not** be resampled;
        resample beforehand if necessary.
    return_probs : bool, default False
        If True, also return the raw soft-max probabilities.

    Returns
    -------
    int
        Detected tone (1-4).  If `return_probs` is True returns
        `(tone, probs)` instead.
    """
    # Build ToneNet input exactly like the training pipeline
    rgb_img = _mel_image(audio, sr=sr)
    net_in = _preprocess_caffe(rgb_img)

    probs = _get_model().predict(net_in[None], verbose=0)[0]
    tone = int(np.argmax(probs) + 1)

    return (tone, probs) if return_probs else tone
