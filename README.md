# MuVidGen – Pixar 3D (Full Feature, Optimized)

**Keeps ALL features** from the Pixar preset:
- Audio analysis (duration, BPM, beats) via librosa
- Lyric analysis (keywords + per-line tags) via NLTK
- HF text-to-image with Streamlit Secrets (`HF_TOKEN`) + model picker
- Character Bible + Negative prompts
- Beat-aware cuts, Ken Burns motion, PIL subtitles
- MP4 preview + Download

**Optimizations to avoid Cloud cold-start hangs**
- Lazy imports for heavy libs (librosa, moviepy)
- NLTK downloads cached with `@st.cache_resource`
- No forced reruns
- Video render `threads=1`

## Streamlit Secrets
Settings → Secrets:
```toml
HF_TOKEN="YOUR_HF_TOKEN"
```
