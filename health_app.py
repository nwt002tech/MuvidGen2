import streamlit as st, sys, platform, numpy
st.set_page_config(page_title="Health", layout="centered")
st.title("MuVidGen Health (No-Pandas build)")
st.write({"python": sys.version, "platform": platform.platform(), "numpy": numpy.__version__})
st.success("If you can see this page, the container boots fine.")
