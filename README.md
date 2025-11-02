# MuVidGen – Pixar 3D (No-Librosa Build, Py3.13-safe)

This build removes `librosa`, `numba`, and `llvmlite` to avoid wheel builds on Streamlit Cloud's Python 3.13.
Audio analysis uses a pure-NumPy onset/energy method (duration, approximate BPM, beat grid).

## Deploy
1) Push to GitHub, set **Main file** to `app.py` on Streamlit Cloud.
2) Add Secrets:
```toml
HF_TOKEN="your_hf_token_here"
```
