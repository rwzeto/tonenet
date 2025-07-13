#!/usr/bin/env python3
"""
detect_cli.py — Minimal CLI wrapper around `detect_tone`.

Usage
-----
• Live mic detection (16 kHz mono):
      $ python detect_cli.py

• Detect tone from an existing audio file:
      $ python detect_cli.py path/to/word.wav
"""

from __future__ import annotations
import argparse, sys
import numpy as np
import sounddevice as sd
import librosa
from tonenet import detect_tone, FS  # assumes detect_tone lives in tonenet.py


# ─────────────── helpers ─────────────── #
def record(fs: int = FS) -> np.ndarray:
    """Record from the default mic until the user presses ⏎ again."""
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


def load_file(path: str, sr: int = FS) -> np.ndarray:
    """Load a mono audio file and normalise."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return librosa.util.normalize(y)


# ─────────────── main ─────────────── #
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mandarin tone detector")
    parser.add_argument("file", nargs="?", help="Optional path to a WAV/FLAC/etc.")
    parser.add_argument(
        "-p",
        "--probs",
        action="store_true",
        help="Print raw soft-max probabilities as well",
    )
    args = parser.parse_args(argv)

    if args.file:
        audio = load_file(args.file)
    else:
        audio = record()

    res = detect_tone(audio, return_probs=args.probs)

    if args.probs:
        tone, probs = res
        print("Raw probabilities:", np.round(probs, 3))
    else:
        tone = res

    print(f"Detected tone → {tone}")


if __name__ == "__main__":
    main()
