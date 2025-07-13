echo "▶ Fetching pre-trained ToneNet weights…"
if [[ ! -f ToneNet.h5 ]]; then
  tmpdir="$(mktemp -d)"
  git clone --depth 1 https://github.com/saber5433/ToneNet.git "$tmpdir"

  # Look for any *.h5 file (there's only one in the repo)
  model_path="$(find "$tmpdir" -type f -iname '*.h5' | head -n 1)"

  if [[ -z "$model_path" ]]; then
    echo "❌ Could not locate an .h5 weight file inside the ToneNet repo." >&2
    echo "   Repo structure may have changed — grab it manually and place it as ToneNet.h5" >&2
    exit 1
  fi

  echo "   Found $(basename "$model_path") — copying to project root."
  cp "$model_path" ToneNet.h5
  rm -rf "$tmpdir"
else
  echo "   ToneNet.h5 already present — skipping download."
fi