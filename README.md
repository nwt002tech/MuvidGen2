# MuVidGen – Pixar 3D (Boot-Safe, FFmpeg-only)

This build is optimized for Streamlit Cloud on Python 3.13:
- **No** `librosa`, `soundfile`, `numba`, `llvmlite`, `nltk`
- Audio decoding via **MoviePy + imageio-ffmpeg** (pure wheels)
- Pure-NumPy beat/BPM detection (short-time energy + onset peaks)
- Pixar-inspired prompt builder, HF txt2img + stylized fallback
- Beat-aware scene cuts, subtitles, MP4 preview & download
- Early “boot beacons” & sidebar checkpoints for debugging

## Deploy
1. Push these files to a repo and set **Main file** → `app.py` in Streamlit Cloud.
2. In **Settings → Secrets**, add:
   ```toml
   HF_TOKEN="hf_...your token..."
   ```
3. (Optional) Clear any old “Python packages” configuration so `requirements.txt` is the only dependency list.

## Usage
- Upload an audio file (mp3/wav/m4a/aac), paste lyrics (1 line = 1 scene).
- Click **Analyze Audio + Lyrics**.
- Click **Generate Full Music Video** to render (24fps, H.264 + AAC).
- If HF is throttled/slow, toggle **Force fallback** to generate stylized cards instead.

## Notes
- The NumPy onset method is fast and robust for pop/children’s music.
- You can later add per-beat hard cuts or duration sliders without new deps.
