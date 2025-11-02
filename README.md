# MuVidGen – Pixar 3D (No-Pandas, FFmpeg-only, Py3.13-safe)

This build removes `pandas` to avoid missing wheels on Python 3.13 in some environments.
All features preserved. Audio via MoviePy+FFmpeg; lyrics keywords via `collections.Counter`.

## Deploy
1) Set **Main file** → `app.py` in Streamlit Cloud.
2) Add secret in Settings → Secrets:
```toml
HF_TOKEN="your_hf_token_here"
```
3) If the app ever "sticks in the oven", temporarily set **Main file** → `health_app.py` to verify the container boots, then switch back.
