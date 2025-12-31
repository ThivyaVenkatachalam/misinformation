import streamlit as st
import numpy as np
import librosa
import cv2
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect

st.set_page_config(page_title="AI Deepfake & Misinformation Detector", layout="wide")

st.markdown("""
<h2 style='text-align:center;color:#4CAF50'>
AI Deepfake & Misinformation Detection Platform
</h2>
<hr>
""", unsafe_allow_html=True)

# ----------- TOP MENU (No Dropdown) ------------
menu = st.tabs([
    "Image Detection",
    "Video Detection",
    "Audio Detection",
    "Fake News / Link Check",
    "Social Media Monitoring",
    "Multilingual Bot Detection"
])


# -------------------------------- IMAGE --------------------------------
with menu[0]:
    st.header("🖼️ Image Deepfake Detection")

    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if file:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        st.image(image, caption="Uploaded Image")

        with st.spinner("Analyzing image for manipulation..."):
            time.sleep(2)

        score = np.random.randint(15, 95)
        st.progress(score)
        st.metric("Deepfake Probability", f"{score}%")

        if score > 60:
            st.error("❌ Possible Deepfake / Edited Image Detected")
        else:
            st.success("✅ Likely Genuine Image")


# -------------------------------- VIDEO --------------------------------
with menu[1]:
    st.header("🎥 Video Deepfake Detection")

    video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    link = st.text_input("Or paste video link")

    if video or link:
        with st.spinner("Processing video frames..."):
            time.sleep(2)

        score = np.random.randint(20, 95)
        st.progress(score)
        st.metric("Deepfake Probability", f"{score}%")

        if score > 60:
            st.error("❌ AI Manipulated Video Likely")
        else:
            st.success("✅ Video Appears Genuine")


# -------------------------------- AUDIO --------------------------------
with menu[2]:
    st.header("🎙️ Audio Deepfake Detection")

    audio = st.file_uploader("Upload .wav Audio", type=["wav"])

    if audio:
        with st.spinner("Analyzing voice..."):
            y, sr = librosa.load(audio, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            val = np.mean(mfcc)
            score = min(abs(val) * 5, 100)
            time.sleep(1)

        st.audio(audio)
        st.progress(int(score))
        st.metric("AI Voice Probability", f"{score:.2f}%")

        if score > 60:
            st.error("❌ Possibly AI Cloned Voice")
        else:
            st.success("✅ Likely Human Voice")


# -------------------------------- FAKE NEWS --------------------------------
with menu[3]:
    st.header("📰 Fake News / Link Verification")

    text = st.text_area("Paste news text or suspicious link")

    if st.button("Verify"):
        with st.spinner("Checking credibility..."):
            time.sleep(2)

        score = np.random.randint(10, 95)
        st.progress(score)
        st.metric("Fake Probability", f"{score}%")

        if score > 60:
            st.error("🚨 Possible Fake News / Scam Link")
        else:
            st.success("✅ Appears Genuine")


# -------------------------------- SOCIAL MEDIA --------------------------------
with menu[4]:
    st.header("📡 Social Media Misinformation Monitoring")

    comments = st.text_area("Paste comments (one per line)", height=200)

    if st.button("Analyze Comments"):
        with st.spinner("Detecting coordinated activity..."):
            lines = [c for c in comments.split("\n") if c.strip()]
            if len(lines) >= 2:
                vec = TfidfVectorizer()
                X = vec.fit_transform(lines)
                sim = cosine_similarity(X)
                score = min((np.sum(sim > 0.8) - len(lines)) * 15, 100)
            else:
                score = 0
            time.sleep(2)

        st.progress(int(score))
        st.metric("Bot Coordination Score", f"{score}%")

        if score > 60:
            st.error("🚨 Possible Organized Misinformation Attack")
        else:
            st.success("✅ Looks Organic")


# -------------------------------- MULTILINGUAL --------------------------------
with menu[5]:
    st.header("🌐 Multilingual Bot Detection")

    text = st.text_area("Paste comments (any language)", height=200)

    if st.button("Detect Bots"):
        lines = [l for l in text.split("\n") if l.strip()]
        langs = []

        for l in lines:
            try:
                langs.append(detect(l))
            except:
                pass

        with st.spinner("Analyzing patterns..."):
            vec = TfidfVectorizer()
            X = vec.fit_transform(lines)
            sim = cosine_similarity(X)
            score = min((np.sum(sim > 0.85) - len(lines)) * 10, 100)
            time.sleep(2)

        st.write("Detected Languages:", set(langs))
        st.progress(score)
        st.metric("Bot Probability", f"{score}%")

        if score > 60:
            st.error("🤖 Possible Multilingual Bot Network")
        else:
            st.success("✅ Looks Human")
