import streamlit as st, sys, platform, numpy, PIL, requests
st.title("MuVidGen Healthcheck")
st.write("Python:", sys.version)
st.write("Platform:", platform.platform())
st.write("NumPy:", numpy.__version__)
import importlib
mods = ["moviepy", "imageio_ffmpeg", "soundfile", "librosa", "numba", "llvmlite", "nltk"]
status = {}
for m in mods:
    try:
        importlib.import_module(m)
        status[m] = "✅"
    except Exception as e:
        status[m] = f"❌ {e}"
st.json(status)
st.success("If you can see ✅ for all modules, the environment is good.")

st.write("Secrets present:", "HF_TOKEN" in st.secrets)
st.write("Done.")
