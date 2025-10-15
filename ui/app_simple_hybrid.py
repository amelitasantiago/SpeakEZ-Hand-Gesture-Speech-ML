# =============================
# ui/app_hybrid_simple.py (FIXED v1.1)
# =============================
# FIX: Added hand_present gate to letter detection path
# FIX: Enhanced initialization logging
# FIX: Improved hysteresis for static gestures
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import os, sys, time, yaml
from collections import deque, defaultdict
import tensorflow as tf
print(f"[SYS] TF {tf.__version__} | Eager: {tf.executing_eagerly()}")

# Load config.yaml (fallback to env)
config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"[SYS] Loaded config.yaml")
else:
    config = {}

# ensure src/ import
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURR_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import cv2
import numpy as np
import joblib

# ---- TTS ----
try:
    from tts import tts_speak_live as _speak
except Exception:
    def _speak(txt: str):
        print(f"[TTS] {txt}")

# ---- Letters ----
try:
    from inference_ import predict_letter, extract_word_features
except Exception as e:
    print(f"[DET][ERR] inference_ import failed: {e}")
    predict_letter = None
    extract_word_features = None

# ---- Word model wiring ----
_WORD_MODEL = None
_WORD_LABELS = None
_WORD_FE_FN = extract_word_features

WORD_MODEL_PATH = os.environ.get("WORD_MODEL_PATH", r"C:\Users\ameli\speakez\models\final\word_skel_logreg_8.joblib")
WORD_PROBA_THRESH = float(os.environ.get("WORD_PROBA_THRESH", config.get('ui', {}).get('prob_threshold_default', 0.70)))
WORD_MIN_HOLD_FRAMES = int(os.environ.get("WORD_MIN_HOLD_FRAMES", 6))
WORD_REPEAT_COOLDOWN_S = float(os.environ.get("WORD_REPEAT_COOLDOWN_S", 1.5))

# ---- Letter stability / cooldown ----
LETTER_MAJORITY_WINDOW = int(os.environ.get("LETTER_MAJORITY_WINDOW", 5))
LETTER_REPEAT_COOLDOWN_S = float(os.environ.get("LETTER_REPEAT_COOLDOWN_S", 0.9))
CONF_THRESH = float(os.environ.get("CONF_THRESH", config.get('ui', {}).get('letter_conf_threshold', 0.80)))

# ---- Hand gate (adjusted for better sensitivity) ----
HAND_MOTION_THR = float(os.environ.get("HAND_MOTION_THR", 0.015))  # Lowered from 0.02
HAND_SKIN_THR   = float(os.environ.get("HAND_SKIN_THR", 0.05))    # Lowered from 0.06
HAND_WARMUP     = int(os.environ.get("HAND_WARMUP", 30))          # Reduced from 60

# ---- Capture ----
CAM_INDEX = int(os.environ.get("CAM_INDEX", 0))
CAP_BACKEND = cv2.CAP_DSHOW  # Windows-friendly
TARGET_W, TARGET_H = 640, 480
TARGET_FPS = int(os.environ.get("TARGET_FPS", 30))

# ---- Logging ----
LOG_EVERY_S = float(os.environ.get("LOG_EVERY_S", 1.0))

# ---- Blank/freeze guards ----
_FRAME_BLANK_STD = float(os.environ.get("FRAME_BLANK_STD", 2.5))
_FRAME_FREEZE_EPS = float(os.environ.get("FRAME_FREEZE_EPS", 0.5))
_FRAME_FREEZE_MAX = int(os.environ.get("FRAME_FREEZE_MAX", 90))

# Utils
def _skin_mask_ycrcb(frame_bgr):
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    return cv2.inRange(ycrcb, lower, upper)

def _ratio(mask):
    return np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])

def _motion_ratio(gray, prev_gray):
    if prev_gray is None:
        return 0.0
    diff = cv2.absdiff(gray, prev_gray)
    return np.mean(diff) / 255.0

def _open_cam():
    print(f"[CAM] Attempting to open camera index {CAM_INDEX} with backend {CAP_BACKEND}")
    cap = cv2.VideoCapture(CAM_INDEX, CAP_BACKEND)
    
    if not cap.isOpened():
        print(f"[CAM][ERR] Failed with CAP_DSHOW, trying default backend...")
        cap = cv2.VideoCapture(CAM_INDEX)
    
    if not cap.isOpened():
        raise RuntimeError(f"[CAM][FATAL] Cannot open camera at index {CAM_INDEX}")
    
    # Set properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    
    # Verify settings
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"[CAM] ✓ Opened successfully: {actual_w}x{actual_h} @ {actual_fps}fps")
    
    # Test read
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        raise RuntimeError("[CAM][FATAL] Camera opened but cannot read frames")
    
    print(f"[CAM] ✓ Test frame captured: shape={test_frame.shape}")
    return cap

# Load word model
try:
    _WORD_MODEL = joblib.load(WORD_MODEL_PATH)
    _WORD_LABELS = _WORD_MODEL.classes_ if hasattr(_WORD_MODEL, 'classes_') else [0,1,2,3,4,5,6,7]
    print(f"[WORD] Loaded model: {WORD_MODEL_PATH} with labels: {_WORD_LABELS}")
    if _WORD_FE_FN:
        print(f"[WORD] Using feature extractor: extract_word_features")
except Exception as e:
    print(f"[WORD][WARN] Model load failed: {e}")
    _WORD_MODEL = None

def main():
    cap = None
    try:
        cap = _open_cam()
    except RuntimeError as e:
        print(f"[SYS][FATAL] {e}")
        return

    # Buffers/state
    letter_window = deque(maxlen=LETTER_MAJORITY_WINDOW)
    word_prob_history = deque(maxlen=WORD_MIN_HOLD_FRAMES * 2)
    last_letter_said_at = defaultdict(float)
    last_word_said_at = defaultdict(float)
    prev_gray = None
    prev_frame = None
    frame_id = 0
    last_log_frame = 0
    last_nonzero_frame = 0
    prev_motion = 0.0
    start_time = time.time()
    
    # Hand persistence tracking (debounce)
    hand_present_frames = 0
    HAND_MIN_FRAMES = 3  # Require 3 consecutive frames before triggering

    # Word label map (natural speech)
    label_map = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven'}

    print(f"[SYS] Starting main loop (press 'q' or ESC to exit)")
    print(f"[SYS] Hand detection: motion>{HAND_MOTION_THR:.3f}, skin>{HAND_SKIN_THR:.3f}, warmup>{HAND_WARMUP}f")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[CAM] End of stream or read error")
            break
        frame_id += 1

        # Blank frame guard
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.std(gray) < _FRAME_BLANK_STD:
            if frame_id % 30 == 0:  # Reduce spam
                print(f"Frame {frame_id}: BLANK (std={np.std(gray):.2f}) | Skipping")
            continue

        # Skin/motion
        try:
            skin_mask = _skin_mask_ycrcb(frame)
        except Exception as e:
            print(f"[HAND][ERR] Skin mask failed: {e}")
            skin_mask = np.zeros((frame.shape[:2]), dtype=np.uint8)
        
        skin_ratio = _ratio(skin_mask)
        motion_ratio = _motion_ratio(gray, prev_gray)

        # Improved hysteresis: Boost motion signal if skin is present
        # This helps with static hand poses (e.g., holding 'O' or 'N')
        adj_motion = motion_ratio + (skin_ratio * 0.02)  # Increased boost
        
        # Raw hand detection (before debounce)
        hand_detected_raw = (
            (adj_motion > HAND_MOTION_THR) and 
            (skin_ratio > HAND_SKIN_THR) and 
            (frame_id > HAND_WARMUP)
        )
        
        # Debounce: Require consecutive frames
        if hand_detected_raw:
            hand_present_frames += 1
        else:
            hand_present_frames = 0
        
        # Only set hand_present after minimum frames
        hand_present = hand_present_frames >= HAND_MIN_FRAMES

        # Trackers (non-absorbing)
        if motion_ratio > 0.001:
            last_nonzero_frame = frame_id
        prev_motion = motion_ratio

        # Freeze guard (escapes on resume)
        is_frozen = (
            abs(motion_ratio - prev_motion) < _FRAME_FREEZE_EPS and 
            (frame_id - last_nonzero_frame) > _FRAME_FREEZE_MAX
        )
        
        if is_frozen:
            if frame_id % 30 == 0:
                print(f"Frame {frame_id}: FROZEN | m={motion_ratio:.3f} s={skin_ratio:.3f}")
            if len(letter_window) > 0:
                letter_window.clear()
            cv2.imshow("SpeakEZ", frame)
            cv2.waitKey(10)
            continue

        fired = False
        
        # Log cadence
        if frame_id - last_log_frame >= int(LOG_EVERY_S * TARGET_FPS):
            fps = frame_id / (time.time() - start_time)
            buf_str = ''.join(letter_window) if letter_window else ''
            hand_str = f"Hand:{hand_present}({hand_present_frames}f)"
            print(f"Frame {frame_id}: FPS={fps:.1f} | {hand_str} | m={motion_ratio:.3f}+{skin_ratio*0.02:.3f}={adj_motion:.3f} s={skin_ratio:.3f} | Buffer:'{buf_str}'")
            last_log_frame = frame_id

        # ----- WORD DETECTION (with hand gate) -----
        if hand_present and _WORD_MODEL is not None and _WORD_FE_FN is not None:
            try:
                feats = np.asarray(_WORD_FE_FN(frame)).reshape(1, -1)
                if hasattr(_WORD_MODEL, 'predict_proba'):
                    proba = _WORD_MODEL.predict_proba(feats)[0]
                    scores = {_WORD_LABELS[i]: float(proba[i]) for i in range(len(_WORD_LABELS))}
                else:
                    d = _WORD_MODEL.decision_function(feats)
                    if d.ndim == 1:
                        p1 = 1.0 / (1.0 + np.exp(-d))
                        scores = {_WORD_LABELS[1]: float(p1), _WORD_LABELS[0]: float(1-p1)}
                    else:
                        expd = np.exp(d - d.max())
                        soft = expd / (expd.sum() + 1e-6)
                        scores = {_WORD_LABELS[i]: float(soft[0,i]) for i in range(len(_WORD_LABELS))}
                
                word_prob_history.append(scores)
                top_word, top_word_p = max(scores.items(), key=lambda kv: kv[1])

                def word_is_stable(hist, w, thr, k):
                    if not hist or len(hist) < k:
                        return False
                    return all(d.get(w, 0.0) >= thr for d in list(hist)[-k:])

                now = time.time()
                if top_word and word_is_stable(word_prob_history, top_word, WORD_PROBA_THRESH, WORD_MIN_HOLD_FRAMES):
                    if now - last_word_said_at[top_word] > WORD_REPEAT_COOLDOWN_S:
                        speak_text = label_map.get(top_word, str(top_word))
                        print(f"[WORD] Top={top_word} p={top_word_p:.2f} hold={WORD_MIN_HOLD_FRAMES}/{WORD_MIN_HOLD_FRAMES} → SPEAK '{speak_text}'")
                        _speak(speak_text)
                        last_word_said_at[top_word] = now
                        fired = True
            except Exception as e:
                print(f"[WORD][ERR] Pipeline failed: {e}")

        # ----- LETTER DETECTION (STRICT HAND GATE) -----
        # Clear buffer immediately if hand is not present
        if not hand_present:
            if len(letter_window) > 0:
                letter_window.clear()
        # Only process letters if hand is present AND no word fired
        elif not fired and predict_letter is not None:
            try:
                letter, conf = predict_letter(frame)
                if letter and conf is not None and letter != "unknown":
                    letter_window.append(letter)
                    # STRICT: Double-check hand is still present before speaking
                    if len(letter_window) >= LETTER_MAJORITY_WINDOW and hand_present:
                        win = list(letter_window)[-LETTER_MAJORITY_WINDOW:]
                        common = max(set(win), key=win.count)
                        if common == letter and conf >= CONF_THRESH:
                            now = time.time()
                            if now - last_letter_said_at[common] > LETTER_REPEAT_COOLDOWN_S:
                                print(f"[DET] Letter='{common}' win({LETTER_MAJORITY_WINDOW}): {','.join(win)} conf={conf:.2f} → SPEAK")
                                _speak(common)
                                last_letter_said_at[common] = now
                else:
                    # Clear stale buffer if detection fails
                    if len(letter_window) > 0:
                        letter_window.popleft()
            except Exception as e:
                print(f"[DET][ERR] predict_letter failed: {e}")

        prev_gray = gray.copy()
        prev_frame = frame.copy()

        # Preview with hand indicator
        display = frame.copy()
        status_color = (0, 255, 0) if hand_present else (0, 0, 255)
        hand_text = f"Hand: {hand_present} ({hand_present_frames}/{HAND_MIN_FRAMES}f)"
        cv2.putText(display, hand_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(display, f"Motion: {adj_motion:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, f"Skin: {skin_ratio:.3f}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("SpeakEZ", display)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

        time.sleep(0.001)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        print("\n[SYS] Interrupted by user; draining TTS queue...")
        try:
            from tts import _get_speaker
            _get_speaker().close(timeout=2.0)
        except:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[SYS] Clean exit.")