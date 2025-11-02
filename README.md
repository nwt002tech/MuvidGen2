# MuVidGen – Free, Mobile-Friendly AI Music Video Generator (Streamlit)

**What it does**
- Upload a song (MP3/WAV/M4A), paste lyrics, pick a style.
- App analyzes tempo/beats/duration + lyric themes.
- Auto-storyboards & generates per-line scene images (Hugging Face token optional, free).
- Assembles a full MP4 music video synced to your song, with subtitles.
- Download the MP4.

**Free stack**
- Hosting: Streamlit Community Cloud (free)
- AI images: Hugging Face Inference API (free token), or fallback text-card visuals
- Video: MoviePy + bundled ffmpeg (imageio-ffmpeg)
- Audio/lyrics analysis: librosa, NLTK/TextBlob, scikit-learn

## Deploy (Community Cloud)
1. Push this folder to a public GitHub repo.
2. Create a free account at https://streamlit.io/cloud and deploy the repo.
3. (Optional) Add a **Hugging Face** token in the app sidebar. Create one free at https://huggingface.co/settings/tokens  
   - You can use any open image model; defaults target SDXL-Turbo-like endpoints where available.

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Works without any external API keys (fallback visuals). With a HF token, you get AI scene images.
- Designed for mobile: large controls, single-column flow, big buttons.
