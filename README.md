# MuVidGen – Pixar 3D (FFmpeg-only, Py3.13-safe)

This build avoids `librosa`, `numba`, `llvmlite`, and `soundfile`.
Audio is decoded with **MoviePy + imageio-ffmpeg** into NumPy arrays for analysis.
Works with Streamlit Cloud's Python 3.13 "uv" environment.

## Deploy
1) Push to GitHub; set **Main file** → `app.py` on Streamlit Cloud.
2) Add Secrets:
```toml
HF_TOKEN="your_hf_token_here"
```
