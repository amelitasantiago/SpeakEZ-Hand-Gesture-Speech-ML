"""
SpeakEZ - Smart Hybrid Streamlit UI
Complete system with letter + word recognition

Run: streamlit run app_streamlit_hybrid.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import os, cv2, av, tempfile, pandas as pd, time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import numpy as np

# Import detection modules
try:
    from utils import load_config
    from inference_ import predict_letter, LetterBuffer, LETTER_CLASSES
    from word_detector import classify_word_video, WORDS
    from tts import tts_speak_live, tts_to_mp3_bytes, list_voices
except Exception as e:
    st.error(f"Import Error: {e}")
    st.stop()

# Page config
st.set_page_config(
    page_title="SpeakEZ - Smart Hybrid",
    page_icon="🤟",
    layout="wide"
)

# Title
st.title("🤟 SpeakEZ - Smart Hybrid System")
st.markdown("**Intelligent sign language recognition combining letter and word detection**")
st.success("✓ All modules loaded successfully", icon="✅")

# Load config
try:
    cfg = load_config()
except:
    cfg = {
        "ui": {
            "letters_smooth_N": 3,
            "webcam_mirror": False,
            "prob_threshold_default": 0.25,
            "top2_margin_default": 0.05,
            "agreement_default": False
        }
    }

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    st.subheader("Letter Recognition")
    letter_conf = st.slider("Confidence threshold", 0.0, 1.0, 0.20, 0.05,
                            help="Lower = more sensitive")
    smooth_n = st.slider("Stability frames", 1, 10, 3,
                         help="Frames needed to agree")
    mirror_cam = st.checkbox("Mirror camera", value=False)
    
    st.subheader("Word Recognition")
    word_conf = st.slider("Word confidence", 0.0, 1.0, 0.50, 0.05)
    word_margin = st.slider("Top-2 margin", 0.0, 0.5, 0.05, 0.01)
    
    st.divider()
    
    st.subheader("ℹ️ System Info")
    st.write(f"📝 Letter classes: {len(LETTER_CLASSES)}")
    st.write(f"✋ Word classes: {len(WORDS)}")
    st.caption(f"Words: {', '.join(WORDS[:4])}...")
    
    st.divider()
    
    st.subheader("📊 Session Stats")
    if 'stats' not in st.session_state:
        st.session_state.stats = {
            "frames": 0,
            "letter_preds": 0,
            "word_preds": 0
        }
    
    st.metric("Frames processed", st.session_state.stats["frames"])
    st.metric("Letter predictions", st.session_state.stats["letter_preds"])
    st.metric("Word predictions", st.session_state.stats["word_preds"])

# Initialize session state
if 'letter_buffer' not in st.session_state:
    st.session_state.letter_buffer = LetterBuffer(N=smooth_n)

if 'word_history' not in st.session_state:
    st.session_state.word_history = []

letters_buffer = st.session_state.letter_buffer

# WebRTC configuration - Fixed for codec issues
rtc_cfg = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
    "sdpSemantics": "unified-plan"
})

# Main tabs
tab1, tab2, tab3 = st.tabs(["📹 Live Recognition", "🎬 Word Upload", "📖 Help"])

# ============================================================================
# TAB 1: Live Letter Recognition
# ============================================================================
with tab1:
    st.markdown("### 📹 Real-time Letter Recognition")
    
    col_info, col_mode = st.columns([3, 1])
    with col_info:
        st.info("👉 Show hand signs to spell words letter-by-letter. System stabilizes over multiple frames.")
    with col_mode:
        mode_indicator = st.empty()
        mode_indicator.success("Mode: **Letter**", icon="📝")
    
    def video_frame_callback(frame):
        """Process each frame for letter detection"""
        try:
            st.session_state.stats["frames"] += 1
            
            img_bgr = frame.to_ndarray(format="bgr24")
        
            if mirror_cam:
                img_bgr = cv2.flip(img_bgr, 1)
            
            # Predict letter
            tok, conf = predict_letter(img_bgr, conf_threshold=letter_conf)
            
            # Update buffer
            updated = letters_buffer.push(tok)
            if updated:
                st.session_state.stats["letter_preds"] += 1
            
            # Overlay UI
            h, w = img_bgr.shape[:2]
            
            # Background panel
            cv2.rectangle(img_bgr, (0, 0), (w, 110), (40, 40, 40), -1)
            
            # Prediction with color
            color = (0, 255, 0) if conf > 0.6 else (0, 255, 255) if conf > 0.3 else (0, 165, 255)
            cv2.putText(img_bgr, f"Sign: {tok}", (15, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
            cv2.putText(img_bgr, f"Conf: {conf:.1%}", (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Buffer preview
            buffer_preview = letters_buffer.buf[-35:] if len(letters_buffer.buf) > 35 else letters_buffer.buf
            if buffer_preview:
                cv2.putText(img_bgr, f"Text: {buffer_preview}", (w-450, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")
    
        except Exception as e:
            # Return original frame if processing fails
            return frame
    
    # Layout: Video + Controls
    col_video, col_controls = st.columns([2, 1])
    
    with col_video:
        try:
            webrtc_ctx = webrtc_streamer(
                key="hybrid_letters",
                mode=WebRtcMode.SENDRECV,
                video_frame_callback=video_frame_callback,
                rtc_configuration=rtc_cfg,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            if webrtc_ctx and webrtc_ctx.state.playing:
                st.success("🟢 Camera active - Sign language being recognized")
            else:
                st.warning("⏸️ Click START to begin recognition")
        
        except Exception as e:
            st.error(f"WebRTC Error: {e}")
            st.info("💡 **Alternative:** Use the OpenCV version: `python ui/app_hybrid_simple.py`")
    
    with col_controls:
        st.markdown("#### 📝 Text Buffer")
        
        buffer_text = st.text_area(
            "Recognized text",
            letters_buffer.buf,
            height=180,
            key="buffer_display"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        
        if col_btn1.button("🗑️ Clear", use_container_width=True):
            letters_buffer.deq.clear()
            letters_buffer.last = None
            letters_buffer.buf = ""
            st.rerun()
        
        if col_btn2.button("⌫ Delete", use_container_width=True):
            letters_buffer.buf = letters_buffer.buf[:-1]
            st.rerun()
        
        st.caption(f"📊 {len(letters_buffer.buf)} characters | Stability: {smooth_n} frames")
        
        # Word quick-add
        st.markdown("#### ✋ Quick Add Words")
        st.caption("Detected words will appear here")
        
        if st.session_state.word_history:
            for word_entry in st.session_state.word_history[-3:]:
                if st.button(f"➕ {word_entry['word']} ({word_entry['conf']:.0%})", 
                           use_container_width=True,
                           key=f"add_{word_entry['word']}_{word_entry['time']}"):
                    letters_buffer.buf += word_entry['word'] + " "
                    st.rerun()
    
    st.divider()
    
    # ========================================================================
    # TTS Controls
    # ========================================================================
    st.markdown("### 🔊 Text-to-Speech")
    
    with st.expander("⚙️ Voice Settings", expanded=False):
        voices = list_voices()
        if voices:
            voice_opts = [f"{vid} — {name}" for vid, name in voices]
            chosen = st.selectbox("Voice", ["(default)"] + voice_opts)
            chosen_voice = None if chosen == "(default)" else voices[voice_opts.index(chosen)][0]
        else:
            chosen_voice = None
        
        rate = st.slider("Speed (words/min)", 100, 250, 180, 10)
        vol = st.slider("Volume", 0.0, 1.0, 1.0, 0.05)
    
    col_tts1, col_tts2 = st.columns(2)
    
    with col_tts1:
        if st.button("🔊 Speak Text", use_container_width=True, type="primary"):
            if letters_buffer.buf.strip():
                with st.spinner("Speaking..."):
                    try:
                        tts_speak_live(letters_buffer.buf, rate=rate, 
                                      voice_id=chosen_voice, volume=vol)
                        st.success("✓ Speech complete!")
                    except Exception as e:
                        st.error(f"TTS error: {e}")
            else:
                st.warning("Buffer is empty")
    
    with col_tts2:
        if st.button("💾 Download MP3", use_container_width=True):
            if letters_buffer.buf.strip():
                try:
                    mp3 = tts_to_mp3_bytes(letters_buffer.buf)
                    st.audio(mp3, format="audio/mp3")
                    st.download_button(
                        "📥 Save MP3",
                        mp3,
                        file_name="speakez_text.mp3",
                        mime="audio/mpeg",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"MP3 error: {e}")
            else:
                st.warning("Buffer is empty")

# ============================================================================
# TAB 2: Word Recognition (Video Upload)
# ============================================================================
with tab2:
    st.markdown("### 🎬 Word Recognition from Video")
    st.info("👉 Upload a 1-3 second video of a complete sign language word gesture")
    
    uploaded = st.file_uploader(
        "Choose video file",
        type=["mp4", "mov", "avi", "webm", "mkv"],
        help="Video showing complete word gesture"
    )
    
    col_thr1, col_thr2, col_thr3 = st.columns(3)
    
    with col_thr1:
        word_thr = st.slider("Confidence", 0.0, 1.0, word_conf, 0.01,
                            key="word_thr", help="Minimum confidence to accept")
    with col_thr2:
        margin_thr = st.slider("Margin", 0.0, 1.0, word_margin, 0.01,
                              key="margin_thr", help="Gap between top 2")
    with col_thr3:
        use_agreement = st.checkbox("Prototype check", value=False,
                                    help="Require similarity to prototype")
    
    if st.button("🎯 Recognize Word", use_container_width=True, type="primary"):
        if uploaded is None:
            st.warning("Please upload a video first")
        else:
            try:
                # Save uploaded file
                suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getbuffer())
                    tmp_path = tmp.name
                
                # Recognize
                with st.spinner("🔄 Analyzing video..."):
                    label, conf, probs = classify_word_video(
                        tmp_path,
                        thr=word_thr,
                        margin=margin_thr,
                        agreement=use_agreement
                    )
                
                os.unlink(tmp_path)
                
                # Display results
                if label != "unknown":
                    st.success(f"✅ Recognized: **{label.upper()}** (confidence: {conf:.1%})")
                    
                    # Add to history
                    st.session_state.word_history.append({
                        "word": label.upper(),
                        "conf": conf,
                        "time": time.time()
                    })
                    st.session_state.stats["word_preds"] += 1
                    
                    # Auto-add to buffer
                    if st.checkbox("➕ Auto-add to text buffer", value=True):
                        letters_buffer.buf += label.upper() + " "
                        st.rerun()
                else:
                    st.warning(f"⚠️ Could not recognize (best: {conf:.1%})")
                
                # Probability table
                st.markdown("**📊 All Word Probabilities**")
                df = pd.DataFrame(
                    sorted(probs.items(), key=lambda x: x[1], reverse=True),
                    columns=["Word", "Probability"]
                )
                st.dataframe(
                    df.style.format({"Probability": "{:.1%}"})
                      .background_gradient(cmap='Greens', subset=['Probability']),
                    use_container_width=True
                )
                
                # TTS for word
                col_w1, col_w2 = st.columns(2)
                
                with col_w1:
                    if st.button("🔊 Speak Word", key="speak_word"):
                        try:
                            tts_speak_live(label.upper(), rate=180)
                            st.success("✓ Speaking...")
                        except Exception as e:
                            st.error(f"TTS: {e}")
                
                with col_w2:
                    if st.button("💾 Download Word MP3", key="dl_word"):
                        try:
                            mp3 = tts_to_mp3_bytes(label.upper())
                            st.audio(mp3, format="audio/mp3")
                            st.download_button(
                                "📥 Save",
                                mp3,
                                file_name=f"speakez_{label}.mp3",
                                mime="audio/mpeg"
                            )
                        except Exception as e:
                            st.error(f"MP3: {e}")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                st.error(traceback.format_exc())

# ============================================================================
# TAB 3: Help & Documentation
# ============================================================================
with tab3:
    st.markdown("""
    ## 📖 How to Use SpeakEZ Hybrid System
    
    ### 🎯 System Modes
    
    This hybrid system combines two recognition approaches:
    
    1. **📝 Letter Recognition (Tab 1)** - Real-time webcam
       - Spell words letter-by-letter
       - Continuous recognition
       - Use SPACE for spaces, DEL for backspace
    
    2. **✋ Word Recognition (Tab 2)** - Video upload
       - Recognize complete word gestures
       - Higher accuracy for known words
       - Upload 1-3 second video clips
    
    ### 🔄 Workflow
    
    **For spelling custom words:**
    1. Use Letter Recognition tab
    2. Sign each letter clearly
    3. Hold for 2-3 seconds (stability)
    4. Use SPACE between words
    
    **For known words:**
    1. Record 1-3 second video of complete gesture
    2. Upload to Word Recognition tab
    3. System recognizes and adds to buffer
    4. Can auto-add to text or manually insert
    
    **Best of both worlds:**
    - Use word recognition for common phrases ("HELLO", "THANK YOU")
    - Use letter spelling for names, custom words, etc.
    - Combine both in the text buffer
    
    ### ✋ Recognized Words
    
    The system currently recognizes these words:
    """)
    
    # Display words in a nice grid
    cols = st.columns(4)
    for i, word in enumerate(WORDS):
        with cols[i % 4]:
            st.info(f"**{word.upper()}**")
    
    st.markdown("""
    ### 💡 Tips for Best Results
    
    **Letter Recognition:**
    - 🖐️ **Clear hand signs**: Form letters distinctly
    - ⏱️ **Hold steady**: 2-3 seconds per letter
    - 💡 **Good lighting**: Bright, even illumination
    - 🎯 **Plain background**: Solid colors work best
    - 📏 **Distance**: 1-2 feet from camera
    - 🔄 **Neutral pose**: Return to NOTHING between signs
    
    **Word Recognition:**
    - 🎬 **Complete gesture**: Show full word motion
    - ⏱️ **Duration**: 1-3 seconds optimal
    - 🎥 **Stable camera**: No shaking
    - 👐 **Full hand visible**: Entire hand in frame
    - 💡 **Same tips**: Lighting, background, etc.
    
    ### ⚙️ Adjustable Parameters
    
    **Letter Confidence** (Sidebar)
    - Lower = more sensitive (more predictions, possibly wrong)
    - Higher = more selective (fewer predictions, more accurate)
    - Start at 0.20 and adjust based on results
    
    **Stability Frames**
    - How many consecutive frames must agree
    - Lower = faster updates, less stable
    - Higher = slower updates, more reliable
    
    **Word Confidence**
    - Minimum confidence for word acceptance
    - 0.50 = balanced (recommended)
    - Adjust if getting too many/few detections
    
    ### 🐛 Troubleshooting
    
    **No letter predictions:**
    - Lower confidence threshold
    - Check lighting and hand visibility
    - Ensure MediaPipe detects hand (green outline)
    
    **Wrong letters:**
    - Hold signs more steady
    - Increase stability frames
    - Improve lighting/background
    
    **Word not recognized:**
    - Check video quality and duration
    - Ensure word is in recognized list
    - Lower word confidence threshold
    
    **Camera not starting:**
    - Check browser permissions
    - Try different browser
    - Check firewall/antivirus
    
    ### 📊 Understanding the Stats
    
    **Frames processed**: Total video frames analyzed  
    **Letter predictions**: How many letters detected  
    **Word predictions**: How many words recognized  
    **Detection rate**: Percentage of frames with predictions
    
    ### 🎓 Advanced Features
    
    - **Word history**: Recent word detections appear as quick-add buttons
    - **Buffer management**: Clear, delete, or manual edit
    - **TTS options**: Adjust voice, speed, volume
    - **MP3 export**: Download audio for presentations
    """)
    
    st.divider()
    
    st.markdown("""
    ### 🔧 Technical Details
    
    **Letter Model**
    - Architecture: CNN (Convolutional Neural Network)
    - Input: 128×128 RGB images
    - Classes: 29 (A-Z + DEL + SPACE + NOTHING)
    - Latency: <50ms per frame
    
    **Word Model**
    - Architecture: Logistic Regression on skeleton features
    - Features: MediaPipe hand landmarks (170 features)
    - Classes: 8 words
    - Processing: 1-3 seconds per video
    
    **System Requirements**
    - Python 3.8+
    - Webcam (for letter recognition)
    - Modern browser (for Streamlit)
    - Internet connection (for WebRTC/TTS)
    """)

# Footer
st.divider()
st.caption("SpeakEZ v2.0 - Smart Hybrid System | MSc AI Project 2025")