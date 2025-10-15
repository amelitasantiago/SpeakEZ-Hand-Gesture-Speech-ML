import os, cv2, av, tempfile, pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from utils import load_config
from inference_ import predict_letter, LetterBuffer
from word_detector import classify_word_video

import pandas as pd
import tempfile
from tts import tts_speak_live, tts_to_mp3_bytes, list_voices

st.set_page_config(page_title="SpeakEZ", layout="wide")
st.title("SpeakEZ — Streamlit Demo")

cfg = load_config()
letters_buffer = LetterBuffer(N=int(cfg["ui"]["letters_smooth_N"]))

# STUN for WebRTC
rtc_cfg = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

tabs = st.tabs(["Letters (webcam)", "Word clip (upload)"])

# -------- TAB 1: Letters via webcam --------
with tabs[0]:
    st.write("Use the webcam to stream letters; the debounced buffer builds a word.")
    #mirror = bool(cfg["ui"].get("webcam_mirror", True))
    mirror = bool(cfg["ui"].get("webcam_mirror", False))  # try False first


    def _video_frame_callback(frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        if mirror:
            img_bgr = cv2.flip(img_bgr, 1)
        tok, conf = predict_letter(img_bgr)         # your CNN letters model
        letters_buffer.push(tok)

        # overlay
        cv2.putText(img_bgr, f"{tok} {conf:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(img_bgr, letters_buffer.buf[-48:], (10,65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

    colL, colR = st.columns([2,1], gap="large")
    with colL:
        webrtc_streamer(
            key="letters",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=_video_frame_callback,
            rtc_configuration=rtc_cfg,
            media_stream_constraints={"video": True, "audio": False},
        )
    with colR:
        st.subheader("Text buffer")
        st.text_area("Output", letters_buffer.buf, height=180)
        c1, c2 = st.columns(2)
        if c1.button("Clear"):
            letters_buffer.deq.clear(); letters_buffer.last=None; letters_buffer.buf=""
        if c2.button("Backspace"):
            letters_buffer.buf = letters_buffer.buf[:-1]
        st.caption(f"Stability N={cfg['ui']['letters_smooth_N']} — updates only when last N tokens agree.")

    st.divider()
    st.subheader("Text-to-Speech")

    # Optional: pick a voice (IDs are system-dependent)
    with st.expander("Voice settings (optional)"):
        voices = list_voices()
        voice_opts = [f"{vid} — {name}" for vid, name in voices] if voices else []
        chosen = st.selectbox("Voice", ["(default)"] + voice_opts, index=0)
        chosen_voice = None if chosen == "(default)" else voices[voice_opts.index(chosen)-0][0]
        rate = st.slider("Rate (words/min)", 100, 220, 180, 5)
        vol  = st.slider("Volume", 0.0, 1.0, 1.0, 0.05)

    col1, col2 = st.columns(2)
    if col1.button("🔊 Speak buffer (offline)"):
        if letters_buffer.buf.strip():
            try:
                tts_speak_live(letters_buffer.buf, rate=rate, voice_id=chosen_voice, volume=vol)
                st.info("Speaking via system TTS (pyttsx3)…")
            except Exception as e:
                st.error(f"TTS error: {e}")
        else:
            st.warning("Buffer is empty.")

    if col2.button("⬇️ Download MP3"):
        if letters_buffer.buf.strip():
            try:
                mp3 = tts_to_mp3_bytes(letters_buffer.buf)
                st.audio(mp3, format="audio/mp3", autoplay=True)
                st.download_button("Save MP3", mp3, file_name="speakez_buffer.mp3", mime="audio/mpeg")
            except Exception as e:
                st.error(f"MP3 error: {e}")
        else:
            st.warning("Buffer is empty.")


# -------- TAB 2: Word recognition by upload --------
with tabs[1]:
    st.write("Upload a short clip (~1s). The word classifier will run on the file.")
    up = st.file_uploader("Video file", type=["mp4","mov","avi","webm","mkv"])

    st.markdown("**Decision thresholds**")
    c1, c2, c3 = st.columns(3)
    thr    = c1.slider("Prob threshold", 0.00, 1.00, float(cfg["ui"]["prob_threshold_default"]), 0.01)
    margin = c2.slider("Top-2 margin", 0.00, 1.00, float(cfg["ui"]["top2_margin_default"]), 0.01)
    agree  = c3.checkbox("Prototype agreement", bool(cfg["ui"]["agreement_default"]))

    if st.button("Recognize"):
        if up is None:
            st.warning("Please upload a video first.")
        else:
            # write to temp path for your pipeline
            suffix = os.path.splitext(up.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
                t.write(up.getbuffer())
                tmp_path = t.name

            label, conf, probs = classify_word_video(tmp_path, thr=thr, margin=margin, agreement=agree)
            os.unlink(tmp_path)

            st.success(f"Prediction: **{label}**  (conf={conf:.2f})")
            df = pd.DataFrame(sorted(probs.items(), key=lambda kv: kv[1], reverse=True),
                              columns=["class","prob"])
            st.dataframe(df.style.format({"prob": "{:.3f}"}), use_container_width=True)

            c1, c2 = st.columns(2)
            if c1.button("🔊 Speak word (offline)"):
                try:
                    tts_speak_live(label, rate=180, voice_id=None, volume=1.0)
                    st.info("Speaking via system TTS…")
                except Exception as e:
                    st.error(f"TTS error: {e}")

            if c2.button("⬇️ Download MP3 of word"):
                try:
                    mp3 = tts_to_mp3_bytes(label)
                    st.audio(mp3, format="audio/mp3", autoplay=True)
                    st.download_button("Save MP3", mp3, file_name=f"speakez_{label}.mp3", mime="audio/mpeg")
                except Exception as e:
                    st.error(f"MP3 error: {e}")

