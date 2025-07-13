#!/usr/bin/env bash
# install_tonet_env.sh — one-shot installer for ToneNet CLI on macOS
#  • Installs Homebrew Python 3.12 if needed
#  • Creates a .venv using that exact interpreter
#  • Installs all Python deps (TensorFlow, librosa, etc.)
#  • Pulls native helpers (ffmpeg, PortAudio) from Homebrew
#  • Verifies that ToneNet.h5 is in place

set -euo pipefail

#───────────── config ─────────────#
PY_VER="3.12"
PY_FORMULA="python@${PY_VER}"
VENV_DIR=".venv"
WEIGHTS_FILE="ToneNet.h5"
# Optional public mirror for the weights; uncomment if you have a URL
# WEIGHTS_URL="https://example.com/path/to/${WEIGHTS_FILE}"
#──────────────────────────────────#

echo "▶️  Checking Homebrew …"
/opt/homebrew/bin/brew --version >/dev/null 2>&1 || {
  echo "❌ Homebrew not found. Install it first: https://brew.sh/"
  exit 1
}

echo "▶️  Ensuring ${PY_FORMULA} is installed …"
if ! brew list --formula "${PY_FORMULA}" >/dev/null 2>&1; then
  brew install "${PY_FORMULA}"  # ≈ 90 MB, adds /opt/homebrew/opt/python@3.12/bin/python3.12
fi  #  [oai_citation:0‡Homebrew Formulae](https://formulae.brew.sh/formula/python%403.12?utm_source=chatgpt.com)

# Refresh PATH for this shell
eval "$(/opt/homebrew/bin/brew shellenv)"

PYBIN="$(brew --prefix ${PY_FORMULA})/bin/python${PY_VER}"
echo "   → Using interpreter: ${PYBIN}"
if [[ ! -x "${PYBIN}" ]]; then
  echo "❌ Expected python3.12 binary not found."
  exit 1
fi

echo "▶️  Creating virtual-env (${VENV_DIR}) …"
${PYBIN} -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "▶️  Upgrading pip / wheel …"
python -m pip install --upgrade pip wheel setuptools

echo "▶️  Installing Python libraries …"
python -m pip install \
  numpy \
  sounddevice \
  librosa \
  matplotlib \
  pillow \
  tensorflow-macos \
  tensorflow-metal

echo "▶️  Installing native helpers via Homebrew …"
brew install portaudio ffmpeg >/dev/null # PortAudio for sounddevice; FFmpeg for librosa’s file IO   [oai_citation:1‡Homebrew Formulae](https://formulae.brew.sh/formula/portaudio?utm_source=chatgpt.com) [oai_citation:2‡Homebrew Formulae](https://formulae.brew.sh/formula/ffmpeg?utm_source=chatgpt.com)

echo "▶️  Verifying ToneNet weights …"
if [[ ! -f "${WEIGHTS_FILE}" ]]; then
  echo "   ${WEIGHTS_FILE} not found in $(pwd)"
  # If you have a direct URL uncomment the next two lines:
  # echo "   Downloading pre-trained weights …"
  # curl -L --fail -o "${WEIGHTS_FILE}" "${WEIGHTS_URL}"
  # [ -f "${WEIGHTS_FILE}" ] || { echo "❌ Could not obtain weights."; exit 1; }
  echo "❌ Please place the pre-trained weights file (${WEIGHTS_FILE}) in the project root."
  deactivate
  exit 1
fi

echo "✅  Setup complete!  Activate the env whenever you work on ToneNet:"
echo "    source ${VENV_DIR}/bin/activate"