import os
import io
import re
import uuid
import json
import math
import time
import base64
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    CompositeVideoClip,
    concatenate_videoclips
)
from moviepy.video.fx.all import resize
import librosa
import soundfile as sf
import requests

# ---------- Page & Mobile tweaks ----------
st.set_page_config(page_title="MuVidGen – AI Music Video", layout="wide")
st.markdown("""
<style>
button[kind="primary"] { font-size: 1.05rem; padding: 0.7rem 1rem; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
textarea, input, select { font-size: 1rem !important; }
@media (max-width: 420px) {
  .stActionButton { transform: scale(1.05); }
}
</style>
""", unsafe_allow_html=True)

# ---------- Utilities ----------
def human_seconds(sec: float) -> str:
    m = int(sec // 60)
    s = int(round(sec % 60))
    return f"{m}:{s:02d}"

def ensure_nltk():
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except:
        nltk.download("stopwords", quiet=True)

def extract_keywords(lyrics: str, top_k: int = 12) -> List[str]:
    ensure_nltk()
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    stop = set(stopwords.words("english"))
    words = [w.lower() for w in word_tokenize(lyrics) if re.match(r"[A-Za-z]+$", w)]
    words = [w for w in words if w not in stop and len(w) > 2]
    freq = pd.Series(words).value_counts()
    return freq.head(top_k).index.tolist()

def chunk_lyrics_to_lines(lyrics: str) -> List[str]:
    raw_lines = [ln.strip() for ln in lyrics.split("\n")]
    lines = [ln for ln in raw_lines if ln]
    return lines

@dataclass
class AudioAnalysis:
    duration: float
    tempo_bpm: float
    beat_times: List[float]

def analyze_audio(file_bytes: bytes, sr_target: int = 44100) -> AudioAnalysis:
    with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        y, sr = librosa.load(tmp.name, sr=sr_target, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    tempo_bpm = float(tempo) if not isinstance(tempo, (list, np.ndarray)) else float(tempo[0])
    return AudioAnalysis(duration=duration, tempo_bpm=tempo_bpm, beat_times=beat_times)

@dataclass
class LyricAnalysis:
    lines: List[str]
    keywords: List[str]
    line_tags: List[List[str]]

def analyze_lyrics(lyrics: str) -> LyricAnalysis:
    lines = chunk_lyrics_to_lines(lyrics)
    kw = extract_keywords(lyrics, top_k=14)
    line_tags = []
    for ln in lines:
        low = [w.lower() for w in re.findall(r"[A-Za-z]+", ln)]
        tags = [w for w in low if w in kw]
        if any(w in low for w in ["dance","dancin","dancing","groove","move","twist","spin","clap","jump"]):
            tags.append("dancing")
        if any(w in low for w in ["sun","sunshine","day","bright","yellow"]):
            tags.append("daytime")
        if any(w in low for w in ["moon","night","stars","glow"]):
            tags.append("night")
        line_tags.append(sorted(list(set(tags))))
    return LyricAnalysis(lines=lines, keywords=kw, line_tags=line_tags)

# ---------- Prompting / Storyboard ----------
@dataclass
class StyleSettings:
    visual_style: str
    characters: str
    palette: str
    camera: str
    model_hint: str

def build_prompt(line: str, tags: List[str], global_kw: List[str], style: StyleSettings) -> str:
    kw_part = ", ".join(sorted(list(set((tags or []) + global_kw[:5]))))
    prompt = (
        f"{style.visual_style}, {style.characters}, scene inspired by lyric: “{line}”. "
        f"Motifs: {kw_part}. {style.camera}. Color palette: {style.palette}. "
        f"Child-friendly, upbeat, vibrant, wholesome."
    )
    if style.model_hint:
        prompt += f" [{style.model_hint}]"
    return prompt

# ---------- Image generation (HF) + Fallback ----------
def hf_txt2img(prompt: str, hf_token: str, model_id: str = "stabilityai/sdxl-turbo", width=768, height=512, steps=4) -> Optional[Image.Image]:
    try:
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {"inputs": prompt, "parameters": {"width": width, "height": height, "num_inference_steps": steps}}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200 and r.headers.get("content-type","").startswith("image/"):
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        if r.status_code == 503:
            time.sleep(3)
            r2 = requests.post(url, headers=headers, json=payload, timeout=120)
            if r2.status_code == 200 and r2.headers.get("content-type","").startswith("image/"):
                return Image.open(io.BytesIO(r2.content)).convert("RGB")
        return None
    except Exception:
        return None

def fallback_card(text: str, size=(768, 512)) -> Image.Image:
    w, h = size
    bg = Image.new("RGB", size, (10, 14, 38))
    overlay = Image.new("RGBA", size, (79, 70, 229, 180))
    bg.paste(overlay, (0,0), overlay)
    draw = ImageDraw.Draw(bg)

    def wrap(t, width=38):
        words = t.split()
        lines = []
        cur = []
        for w_ in words:
            cur.append(w_)
            if len(" ".join(cur)) > width:
                lines.append(" ".join(cur))
                cur = []
        if cur: lines.append(" ".join(cur))
        return "\n".join(lines)

    txt = wrap(text, width=40)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
        font2 = ImageFont.truetype("DejaVuSans.ttf", 22)
    except:
        font = ImageFont.load_default()
        font2 = ImageFont.load_default()

    title = "Scene"
    bbox = draw.textbbox((0,0), title, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    draw.text(((w-tw)//2, 24), title, fill=(255,255,255), font=font)

    bx, by = 40, 90
    draw.multiline_text((bx, by), txt, fill=(230, 235, 255), font=font2, spacing=6)
    return bg

# ---------- Subtitle via PIL (no ImageMagick dependency) ----------
def subtitle_clip(text: str, width: int, height: int, duration: float) -> ImageClip:
    pad_h = 20
    box_h = 110
    img = Image.new("RGBA", (width, height), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    # translucent box bottom area
    draw.rectangle([(0, height - box_h - pad_h), (width, height)], fill=(0,0,0,130))
    # wrap text
    words = text.split()
    lines, cur = [], []
    max_chars = max(10, int(width/20))
    for w_ in words:
        cur.append(w_)
        if len(" ".join(cur)) > max_chars:
            lines.append(" ".join(cur))
            cur = []
    if cur: lines.append(" ".join(cur))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 36)
    except:
        font = ImageFont.load_default()

    y = height - box_h - pad_h + 20
    for ln in lines[:3]:
        bbox = draw.textbbox((0,0), ln, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.text(((width - tw)//2, y), ln, font=font, fill=(255,255,255,255))
        y += th + 6

    return ImageClip(np.array(img)).set_duration(duration).set_position(("center","center"))

# ---------- Video assembly ----------
def ken_burns(image: Image.Image, duration: float, zoom: float = 1.1) -> ImageClip:
    clip = ImageClip(np.array(image)).set_duration(duration)
    return clip.fx(resize, lambda t: 1 + (zoom - 1) * (t / duration))

def assemble_video(
    images: List[Image.Image],
    lines: List[str],
    audio_path: str,
    beat_times: List[float],
    total_duration: float,
    scene_pad: float = 0.15
) -> Tuple[str, float]:
    n = max(1, len(lines))
    target_seg = total_duration / n
    width, height = images[0].size
    t_cursor = 0.0
    clips = []
    beat_idx = 0

    for i in range(n):
        seg_end_target = t_cursor + target_seg
        while beat_idx < len(beat_times) and beat_times[beat_idx] < seg_end_target:
            beat_idx += 1
        if beat_idx < len(beat_times):
            seg_end = beat_times[beat_idx]
        else:
            seg_end = seg_end_target
        seg_start = t_cursor
        seg_duration = max(0.6, seg_end - seg_start)

        img_clip = ken_burns(images[i % len(images)], duration=seg_duration)
        sub_clip = subtitle_clip(lines[i], width, height, max(0.6, seg_duration - scene_pad)).set_start(scene_pad/2.0)
        comp = CompositeVideoClip([img_clip, sub_clip], size=(width, height)).set_duration(seg_duration)
        clips.append(comp)
        t_cursor = seg_end

    current_sum = sum([c.duration for c in clips])
    if current_sum < total_duration and clips:
        last = clips[-1]
        pad = total_duration - current_sum
        clips[-1] = last.set_duration(last.duration + pad)

    video = concatenate_videoclips(clips, method="compose")
    audio = AudioFileClip(audio_path).subclip(0, total_duration)
    final = video.set_audio(audio)

    out_path = os.path.join(tempfile.gettempdir(), f"muvidgen_{uuid.uuid4().hex}.mp4")
    final.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac",
                          temp_audiofile=os.path.join(tempfile.gettempdir(), "temp-audio.m4a"),
                          remove_temp=True, threads=2, verbose=False, logger=None)
    final.close()
    video.close()
    audio.close()
    return out_path, total_duration

# ---------- UI ----------
st.title("🎵 MuVidGen — Free AI Music Video (Mobile-Friendly)")

with st.sidebar:
    st.header("Settings")
    st.caption("Works completely free. Add a free Hugging Face token for AI images.")
    hf_token = st.text_input("Hugging Face Token (optional)", type="password", help="Create free at huggingface.co/settings/tokens")
    model_id = st.text_input("HF Image Model ID", value="stabilityai/sdxl-turbo", help="Any text-to-image model with Inference API enabled.")
    st.divider()
    st.subheader("Visual Style")
    visual_style = st.selectbox("Style", [
        "Pixar-inspired 3D, soft lighting, friendly",
        "2D cartoon, bold outlines, bright colors",
        "Paper cut-out, layered textures, playful",
        "Claymation look, soft depth, cozy lighting",
        "Watercolor children’s book, gentle wash"
    ])
    characters = st.text_input("Characters (optional)", value="Happy anthropomorphic letters and kids dancing")
    palette = st.text_input("Color Palette", value="pastels with pops of yellow, teal, magenta")
    camera = st.text_input("Camera Direction", value="gentle dolly and slow zoom-in for energy")
    model_hint = st.text_input("Model Hint (optional)", value="kid-safe, cheerful, vibrant")
    st.divider()
    st.caption("Tip: If you run out of free HF calls, the app falls back to styled text cards so you can still render the full video.")

st.markdown("#### 1) Upload audio and paste lyrics")
audio_file = st.file_uploader("Upload song (MP3/WAV/M4A)", type=["mp3","wav","m4a","aac"])
lyrics = st.text_area("Paste full song lyrics", height=220, placeholder="Verse/Chorus lines on separate lines work best for scene mapping.")

st.markdown("#### 2) Generate")
colA, colB = st.columns(2)
with colA:
    analyze_btn = st.button("Analyze Audio + Lyrics", type="secondary", use_container_width=True)
with colB:
    render_btn = st.button("Generate Full Music Video", type="primary", use_container_width=True, disabled=True)

if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "story" not in st.session_state:
    st.session_state.story = None

if analyze_btn:
    if not audio_file or not lyrics.strip():
        st.error("Please upload an audio file and paste lyrics first.")
    else:
        with st.spinner("Analyzing audio…"):
            audio_bytes = audio_file.read()
            ana = analyze_audio(audio_bytes)
        with st.spinner("Analyzing lyrics…"):
            lyr = analyze_lyrics(lyrics)

        st.session_state.analysis = {
            "duration": ana.duration,
            "tempo_bpm": ana.tempo_bpm,
            "beat_times": ana.beat_times
        }
        st.session_state.story = {
            "lines": lyr.lines,
            "keywords": lyr.keywords,
            "line_tags": lyr.line_tags
        }

        st.success("Analysis complete ✅")
        st.write(f"**Duration:** {human_seconds(ana.duration)}  |  **Tempo:** {round(ana.tempo_bpm)} BPM  |  **Detected beats:** {len(ana.beat_times)}")
        st.write("**Top lyric keywords:**", ", ".join(lyr.keywords))
        st.caption("Next: click **Generate Full Music Video** below.")

        st.session_state.audio_bytes = audio_bytes
        st.session_state.audio_name = audio_file.name

        st.session_state.style = {
            "visual_style": visual_style,
            "characters": characters,
            "palette": palette,
            "camera": camera,
            "model_hint": model_hint
        }

        st.session_state.enable_render = True

if st.session_state.get("enable_render"):
    st.session_state["render_ready"] = True

if st.session_state.get("render_ready"):
    st.session_state["render_ready"] = False
    st.experimental_rerun()

if "render_ready" in st.session_state and not st.session_state["render_ready"]:
    with colB:
        render_btn = st.button("Generate Full Music Video", type="primary", use_container_width=True)

if render_btn:
    if not st.session_state.get("analysis") or not st.session_state.get("story"):
        st.error("Please run analysis first.")
    else:
        style = StyleSettings(**st.session_state.style)
        lines = st.session_state.story["lines"]
        kws = st.session_state.story["keywords"]
        tags = st.session_state.story["line_tags"]
        beat_times = st.session_state.analysis["beat_times"]
        duration = st.session_state.analysis["duration"]

        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(st.session_state.audio_name)[-1], delete=False) as atmp:
            atmp.write(st.session_state.audio_bytes)
            atmp.flush()
            audio_path = atmp.name

        st.markdown("#### 3) Scene Generation")
        st.caption("Creating per-line scenes (AI images if HF token provided; otherwise clean motion cards).")
        gen_progress = st.progress(0)
        images: List[Image.Image] = []
        total = max(1, len(lines))
        for i, line in enumerate(lines):
            prompt = build_prompt(line, tags[i] if i < len(tags) else [], kws, style)
            img = None
            if hf_token.strip():
                img = hf_txt2img(prompt, hf_token.strip(), model_id=model_id.strip() or "stabilityai/sdxl-turbo")
            if img is None:
                img = fallback_card(line)
            images.append(img)
            gen_progress.progress(int(((i+1)/total)*100))
            if (i+1) % max(1, total//3) == 0 or (i+1) == total:
                st.image(img, caption=f"Scene {i+1}: {line[:80]}", use_column_width=True)

        st.success("Scenes ready ✅")

        st.markdown("#### 4) Rendering Video")
        with st.spinner("Assembling video (this may take a minute)…"):
            out_path, vid_dur = assemble_video(images, lines, audio_path, beat_times, total_duration=duration)

        st.success(f"Video ready ✅  (length: {human_seconds(vid_dur)})")
        with open(out_path, "rb") as f:
            video_bytes = f.read()

        st.video(video_bytes)

        st.download_button(
            label="⬇️ Download MP4",
            data=video_bytes,
            file_name="muvidgen_video.mp4",
            mime="video/mp4",
            use_container_width=True
        )

        st.info("Tip: Re-render after tweaking style or adding an HF token for richer visuals.")

st.markdown("---")
st.caption("Built for mobile. 100% free stack. Add your free Hugging Face token for AI images; otherwise it still renders with animated text cards.")
