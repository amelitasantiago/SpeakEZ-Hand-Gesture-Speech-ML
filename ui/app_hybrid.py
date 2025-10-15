# =============================
# SpeakEZ - Hybrid UI (MediaPipe + Your Models)
# =============================
# Adds:
#   - Proper MediaPipe Hands init (named args)
#   - Hand gating via handedness score + bbox area + hysteresis
#   - WaveDetector  "hi"/"hello" from wrist oscillation
#   - Word ML debounced and guarded
#   - TTS multi-backend (tts.py  SAPI  pyttsx3  print)
#   - Patterns include HELLO/HI/THANKYOU (letters fallback)
# =============================
import os, sys, time, json, platform, gc, warnings
from pathlib import Path
from collections import deque, defaultdict, Counter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF/MediaPipe warnings (0=all, 1=info, 2=warning, 3=error)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
LETTER_SPEAK_MAX_MOTION = float(os.environ.get("LETTER_SPEAK_MAX_MOTION", "0.030")) # REM Increased from 0.028 to allow slight motion
BUFFER_VETO_NO = int(os.environ.get("BUFFER_VETO_NO", "1"))
WORD_DEBUG = int(os.environ.get("WORD_DEBUG", "2"))  # 1 to see candidate logs, 0 to hide

import cv2
import numpy as np
import yaml
import tensorflow as tf

print(f"[SYS] TF {tf.__version__} | Eager: {tf.executing_eagerly()}")
try:
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get("TF_NUM_INTRAOP_THREADS", "1")))
    tf.config.threading.set_inter_op_parallelism_threads(int(os.environ.get("TF_NUM_INTEROP_THREADS", "1")))
except Exception:
    pass
try:
    cv2.setNumThreads(int(os.environ.get("OMP_NUM_THREADS", "1")))
except Exception:
    pass

# Suppress protobuf deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

# -------- MediaPipe Hands --------
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
    print("[SYS] MediaPipe available")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[SYS] MediaPipe not available"); sys.exit(1)

# -------- Config (optional) --------
ROOT = Path(__file__).resolve().parents[1]
cfg_path = ROOT / "config" / "config.yaml"
config = {}
if cfg_path.exists():
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    print("[SYS] Loaded config.yaml")

# -------- TTS (robust) --------
def _make_speaker():
    # 1) Your tts.py if it actually plays audio
    try:
        from tts import tts_speak_live as _user_tts
        def _s1(txt):
            try:
                _user_tts(txt); return True
            except Exception:
                return False
        if _s1("ready"): print("[SYS]  TTS loaded (user module)"); return lambda s: _s1(s)
        print("[SYS] User TTS loaded but silent; falling back.")
    except Exception:
        print("[SYS] No user TTS; falling back.")
    # 2) Windows SAPI
    try:
        import win32com.client
        spk = win32com.client.Dispatch("SAPI.SpVoice")
        spk.Volume = int(os.environ.get("TTS_VOLUME", "100"))
        spk.Rate   = int(os.environ.get("TTS_RATE", "0"))
        def _s2(txt): spk.Speak(txt, 1); return True
        print("[SYS] TTS loaded (Windows SAPI)"); return _s2
    except Exception as e:
        print(f"[SYS] SAPI unavailable: {e}")
    # 3) pyttsx3
    try:
        import pyttsx3
        eng = pyttsx3.init()
        eng.setProperty("volume", float(os.environ.get("TTS_VOLUME_PCT", "1.0")))
        eng.setProperty("rate",   int(os.environ.get("TTS_RATE_WPM", "175")))
        def _s3(txt): eng.say(txt); eng.runAndWait(); return True
        print("[SYS] TTS loaded (pyttsx3)"); return _s3
    except Exception as e:
        print(f"[SYS] pyttsx3 unavailable: {e}")
    # 4) Print fallback
    def _s0(txt): print(f"[TTS] {txt}"); return False
    print("[SYS] No audio TTS available; console only."); return _s0
_speak = _make_speaker()

# -------- Inference hooks --------
sys.path.insert(0, str(ROOT / "src"))
try:
    from inference_ import predict_letter, predict_word
    WORD_PRED_OK = True
    print("[SYS] predict_letter(), predict_word() loaded")
except Exception as e:
    print(f"[SYS] word prediction unavailable: {e}")
    from inference_ import predict_letter
    WORD_PRED_OK = False

# -------- Model dir visibility --------
print("\n[MODEL DETECTION]")
models_dir = ROOT / "models" / "final"
if models_dir.exists():
    print(f"  Scanning: {models_dir}")
    for p in sorted(models_dir.glob("*")):
        if p.suffix.lower() in {".h5", ".joblib", ".json"}:
            print(f"Found: {p.name}")
else:
    print(f"Models directory not found: {models_dir}")
print()

# -------- Optional: word whitelist (restrict what the word-ML may speak) --------
WORDS_WHITELIST = set()
ws_path = models_dir / "words_subset.json"
try:
    if ws_path.exists():
        with open(ws_path, "r") as f:
            _ws = json.load(f)
        if isinstance(_ws, dict):
            words_list = _ws.get("words", [])
        else:
            words_list = _ws
        WORDS_WHITELIST = {str(w).strip().lower() for w in words_list if isinstance(w, (str, int))}
        print(f"[SYS] Word whitelist loaded ({len(WORDS_WHITELIST)} items)")
    else:
        print("[SYS](no words_subset.json) allowing all word labels")
except Exception as e:
    print(f"[SYS] words_subset.json load failed: {e}")
    WORDS_WHITELIST = set()

# -------- Class-specific thresholds (optional thresholds.json) --------
LETTER_THRESH = {"default": float(os.environ.get("CONF_THRESH", "0.72"))}
thr_path = models_dir / "thresholds.json"
try:
    if thr_path.exists():
        with open(thr_path, "r") as f:
            LETTER_THRESH.update(json.load(f))
        print(f"[SYS] Loaded per-class thresholds ({thr_path.name})")
except Exception as e:
    print(f"[SYS] thresholds.json load failed: {e}")

def _thr_for(letter: str) -> float:
    return float(LETTER_THRESH.get(letter, LETTER_THRESH.get("default", 0.72)))

# Per-word thresholds pulled from thresholds.json["words"], fallback to env WORD_MIN_CONF
WORD_THRESH = {}
if "words" in LETTER_THRESH:
    WORD_THRESH = LETTER_THRESH["words"]

#def _word_thr_for(word: str) -> float:
    #return float(WORD_THRESH.get(word, float(os.environ.get("WORD_MIN_CONF", "0.80"))))
# Tighten per-word threshold for â€œpleaseâ€
def _word_thr_for(word: str) -> float:
    words_thr = LETTER_THRESH.get("words", {})
    base = float(words_thr.get(word, LETTER_THRESH.get("default", 0.80)))
    # make 'please' harder to trigger unless the model is truly confident
    if word == "please":
        base = max(base, 0.93)
    return base


# -------- Tunables --------
ENABLE_LETTER_DETECTION = True
ENABLE_WORD_DETECTION   = bool(int(os.environ.get("ENABLE_WORD_DETECTION", "1")))
ENABLE_PATTERN_MATCHING = bool(int(os.environ.get("ENABLE_PATTERN_MATCHING", "1")))
ENABLE_CLAHE_ROI        = bool(int(os.environ.get("ENABLE_CLAHE_ROI", "1")))

# fixed for buffer clearing;later set CLEAR_BUFFER_ON_NO_HAND=0 if you prefer to keep the buffer across very brief dropouts
BUFFER_IDLE_CLEAR_S = float(os.environ.get("BUFFER_IDLE_CLEAR_S", "4.0")) # was 1.2
CLEAR_BUFFER_ON_NO_HAND = bool(int(os.environ.get("CLEAR_BUFFER_ON_NO_HAND", "1")))

CONF_THRESH = float(os.environ.get("CONF_THRESH", "0.72"))
LETTER_MAJORITY_WINDOW = int(os.environ.get("LETTER_MAJORITY_WINDOW", "12"))
LETTER_REPEAT_COOLDOWN_S = float(os.environ.get("LETTER_REPEAT_COOLDOWN_S", "2.8"))
GLOBAL_MIN_DETECTION_INTERVAL_S = float(os.environ.get("GLOBAL_MIN_DET_INT_S", "0.9"))

PATTERN_MATCH_LENGTH = int(os.environ.get("PATTERN_MATCH_LENGTH", "12"))
WORD_COOLDOWN_S      = float(os.environ.get("WORD_COOLDOWN_S", "4.0"))
WORD_ML_COOLDOWN_S   = float(os.environ.get("WORD_ML_COOLDOWN_S", "6.0"))  # Increased to reduce repeats


INFER_EVERY = int(os.environ.get("INFER_EVERY", "2"))

CAM_INDEX = int(os.environ.get("CAM_INDEX", "0"))
TARGET_W, TARGET_H, TARGET_FPS = 640, 480, 30
DEBUG_VISUAL = True
SHOW_CONFIDENCE_BARS = True

# Hand gating / hysteresis
HAND_ON_FRAMES  = int(os.environ.get("HAND_ON_FRAMES", "2"))
HAND_OFF_FRAMES = int(os.environ.get("HAND_OFF_FRAMES", "8")) # make the disarm hysteresis slightly stronger:from 6 to 8
MIN_BBOX_FRAC   = float(os.environ.get("MIN_BBOX_FRAC", "0.06"))
MIN_HAND_SCORE  = float(os.environ.get("MIN_HAND_SCORE", "0.50"))

# Make word-ML actually speak: Add tunables with the others (Tunables section):
WORD_MIN_CONF    = float(os.environ.get("WORD_MIN_CONF", "0.80"))  # Raised default
WORD_MIN_SCORE   = float(os.environ.get("WORD_MIN_SCORE", "0.80")) # was 85
WORD_MIN_AREA    = float(os.environ.get("WORD_MIN_AREA",  "0.075"))  # was .08 Raised to filter small/noise

# Add a few lightweight gates (open palm, skin, motion band, hand streak)
WORD_OPEN_FRAC_MIN  = float(os.environ.get("WORD_OPEN_FRAC_MIN", "0.55"))  # more open hand
WORD_SKIN_MIN_FRAC  = float(os.environ.get("WORD_SKIN_MIN_FRAC", "0.05"))  # visible skin in ROI
WORD_MOTION_STD_MIN = float(os.environ.get("WORD_MOTION_STD_MIN", "0.0015"))
WORD_MOTION_STD_MAX = float(os.environ.get("WORD_MOTION_STD_MAX", "0.012"))
WORD_HAND_STREAK_ON = int(os.environ.get("WORD_HAND_STREAK_ON", "2"))      # was 3 consecutive frames with a good hand
WORD_STABLE_FRAMES  = int(os.environ.get("WORD_STABLE_FRAMES", "3"))       # votes needed before speaking
WORD_VOTES_K        = int(os.environ.get("WORD_VOTES_K", "6"))             # vote window

# Minimum skin mask fraction for valid word prediction (new)
MIN_SKIN_FRAC = float(os.environ.get("MIN_SKIN_FRAC", "0.15"))

# LetterWord pattern fallback (compressed tail)
WORD_PATTERNS = {
    # Core requested:
    "HELLO": "hello",
    "HI": "hi",
    "THANKYOU": "thank you",
    "THANKU": "thank you",
    "TQ": "thank you",
    # Keep small set:
    "NO":"no","NOO":"no","NOOO":"no",
    "ON":"on","OF":"off","OFF":"off",
    "HELP":"help","OK":"okay","YES":"yes"
}

# -------- Utils --------
def _open_cam():
    candidates = list(range(CAM_INDEX, CAM_INDEX + 6))
    if platform.system().lower().startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    else:
        backends = [0]
    for idx in candidates:
        for be in backends:
            cap = cv2.VideoCapture(idx, be) if be != 0 else cv2.VideoCapture(idx)
            if not cap.isOpened(): cap.release(); continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
            cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"[CAM] Opened index {idx} using backend {be} | shape={frame.shape}")
                return cap
            cap.release()
    raise RuntimeError("Cannot open camera. Close other apps and check Windows camera privacy settings.")

def _hand_bbox_from_landmarks(landmarks, w, h, pad=0.35):
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    x0 = max(min(xs), 0.0); y0 = max(min(ys), 0.0)
    x1 = min(max(xs), 1.0); y1 = min(max(ys), 1.0)
    cx = (x0 + x1) / 2; cy = (y0 + y1) / 2
    side = max((x1 - x0), (y1 - y0)) * (1.0 + pad)
    x0 = int(max((cx - side/2) * w, 0)); y0 = int(max((cy - side/2) * h, 0))
    x1 = int(min((cx + side/2) * w, w)); y1 = int(min((cy + side/2) * h, h))
    return x0, y0, x1, y1

def _crop_hand_roi_with_rect(frame_bgr, rect):
    if rect is None: return None
    x0,y0,x1,y1 = rect
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0: return None
    h, w = roi.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), roi.dtype)
    y_off = (side - h) // 2; x_off = (side - w) // 2
    canvas[y_off:y_off+h, x_off:x_off+w] = roi
    return canvas

def _hand_quality(results, W, H):
    if not results.multi_hand_landmarks:
        return 0.0, 0.0, None
    lms = results.multi_hand_landmarks[0]
    score = 1.0
    try:
        if hasattr(results, "multi_handedness") and results.multi_handedness:
            score = float(results.multi_handedness[0].classification[0].score)
    except Exception:
        score = 1.0
    x0,y0,x1,y1 = _hand_bbox_from_landmarks(lms, W, H, pad=0.35)
    area = (x1-x0)*(y1-y0) / float(max(1, W*H))
    return score, area, (x0,y0,x1,y1)

def _fingers_extended_fraction(lms) -> float:
    """
    Returns fraction in [0,1] of the 4 non-thumb fingers that look extended,
    using a simple PIP TIP vs PIP MCP collinearity test in 3D.
    """
    import numpy as _np

    def _vec(a, b):
        return _np.array([a.x - b.x, a.y - b.y, (a.z - b.z)], dtype=_np.float32)

    def _cos(a, b):
        na = _np.linalg.norm(a) + 1e-6
        nb = _np.linalg.norm(b) + 1e-6
        return _np.dot(a, b) / (na * nb)

    finger_ids = [(8,6,5), (12,10,9), (16,14,13), (20,18,17)]  # TIP, PIP, MCP for index/middle/ring/pinky
    extended = 0
    for tip, pip, mcp in finger_ids:
        v1 = _vec(lms.landmark[tip], lms.landmark[pip])
        v2 = _vec(lms.landmark[pip], lms.landmark[mcp])
        cosine = _cos(v1, v2)
        if cosine > 0.85:  # Near collinear (extended)
            extended += 1
    return extended / 4.0

def _pre_enhance(roi_bgr, clip=2.0, tile=8):
    """
    CLAHE on luminance for better contrast (pre-inference).
    """
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile,tile))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def draw_conf_bars(img, preds, x0=10, y0=120, bar_h=8, bar_w=100, font_scale=0.45):
    """
    Draws horizontal confidence bars for each predicted class/conf pair.
    """
    for i, (label, conf) in enumerate(preds):
        y = y0 + i * 25
        cv2.rectangle(img, (x0, y), (x0 + int(bar_w * conf), y + bar_h), (0,255,0), -1)
        cv2.putText(img, f"{label}: {conf:.2f}", (x0 + 5, y + bar_h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1)

class WaveDetector:
    """
    Detects lateral wrist oscillation for "hi/hello" wave.
    Tracks wrist x-pos over window, counts sign-flips in velocity, measures amplitude.
    """
    def __init__(self, window_size=20, min_flips=3, min_amp=0.06, veto_std=0.015, lockout_s=3.0):
        self.window = deque(maxlen=window_size)
        self.motion_std_veto = veto_std  # Motion veto threshold for letter/word inference
        self.min_flips = min_flips
        self.min_amp = min_amp
        self.lockout_s = lockout_s
        self.last_spoken = 0.0
        self.last_amp = 0.0
        self.last_flips = 0
        self.last_open_frac = 0.0

    def add(self, wrist_x: float, open_frac: float):
        self.window.append(wrist_x)
        self.last_open_frac = open_frac

    def motion_level(self) -> float:
        if len(self.window) < 3: return 0.0
        diffs = np.diff(list(self.window))
        return float(np.std(diffs))

    def is_wave(self) -> bool:
        if len(self.window) < self.window.maxlen: return False
        vel = np.diff(list(self.window))
        flips = np.sum(vel[:-1] * vel[1:] < 0)
        amp = float(np.ptp(self.window))
        self.last_flips = flips
        self.last_amp = amp
        return (flips >= self.min_flips and amp >= self.min_amp and self.last_open_frac >= 0.55)

    def is_active(self, now: float) -> bool:
        return (now - self.last_spoken) < self.lockout_s

# -------- Compression for pattern matching --------
def compress_buffer(buf: list[str]) -> str:
    """
    Compress repeated letters to single (for pattern matching).
    """
    seq = list(buf)               # <-- handles deque safely
    if not seq:
        return ""
    out = [seq[0]]
    for c in seq[1:]:
        if c != out[-1]:
            out.append(c)
    return "".join(out)

def check_pattern(buf: list[str], patterns: dict[str,str], min_len: int) -> tuple[str,str,str]:
    """
    Checks compressed buffer tail for patterns.
    Returns (matched_pattern, spoken_word, full_compressed) or ("","", "")
    """
    if len(buf) < min_len: return "","", ""
    comp = compress_buffer(buf)
    if len(comp) < 2: return "","", comp
    for plen in range(min(8, len(comp)), 1, -1):
        tail = comp[-plen:]
        if tail in patterns:
            return tail, patterns[tail], comp
    return "","", comp

# -------- Main --------
def main():
    print("="*70)
    print("HYBRID MODE: MediaPipe + Models (letters, words, patterns, wave)")
    print("="*70)
    print(f"Conf>{CONF_THRESH:.2f}  Window={LETTER_MAJORITY_WINDOW}  INFER_EVERY={INFER_EVERY}")
    print(f"Patterns: {'ON' if ENABLE_PATTERN_MATCHING else 'OFF'}  Word-ML: {'ON' if ENABLE_WORD_DETECTION else 'OFF'}  Wave: ON")
    print("Press 'q' to quit, 's' to save screenshot")
    print("="*70 + "\n")

    hands = mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1,
        static_image_mode=False,
    )

    cap = _open_cam()
    _, frame = cap.read()
    H, W = frame.shape[:2]

    # Hysteresis counters
    hand_on_count = 0
    hand_off_count = 0
    hand_present = False

    # Buffers & state
    letter_window = deque(maxlen=60)  # Sliding window for majority vote
    last_compressed_tail = ""
    last_top = None
    same_top_streak = 0
    last_letter_said_at = defaultdict(float)
    last_word_spoken_at = 0.0
    last_any_detection_at = 0.0
    last_word_ml_at = 0.0
    last_gc = 0
    GC_EVERY = 100

    # Counters
    frame_id = 0
    start = time.time()
    total_L = total_P = total_R = 0
    pred_hist = Counter()
    word_hist = Counter()

    # Wave detector
    wave = WaveDetector(window_size=25, min_flips=4, min_amp=0.07, veto_std=0.012, lockout_s=4.0)

    while cap.isOpened():
        roi = None  # Reset ROI each frame to prevent carry-over
        rect = None  # Reset rect each frame
        score = 0.0
        area = 0.0
        fired = False
        curr_open_frac = 0.0
        curr_preds = [("?", 0.0)]

        ok, frame = cap.read()
        if not ok: break
        frame_id += 1

        # MediaPipe process
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        # Hand quality & gating
        score, area, rect = _hand_quality(results, W, H)
        hand_on = (score >= MIN_HAND_SCORE and area >= MIN_BBOX_FRAC)

        if hand_on:
            hand_on_count += 1
            hand_off_count = 0
            if hand_on_count >= HAND_ON_FRAMES:
                hand_present = True
        else:
            hand_off_count += 1
            hand_on_count = 0
            if hand_off_count >= HAND_OFF_FRAMES:
                hand_present = False

        # Clear buffer on prolonged no-hand (idle)
        if CLEAR_BUFFER_ON_NO_HAND and not hand_present and len(letter_window) > 0:
            #idle_s = time.time() - max(last_letter_said_at.values(), last_word_spoken_at, last_word_ml_at, 0)
            # compute the most recent event timestamp (letters or words), robustly
            _base_times = [last_word_spoken_at, last_word_ml_at, 0.0]
            if last_letter_said_at:  # only extend if we have any letter timestamps
                _base_times.extend(last_letter_said_at.values())
            _last_event_ts = max(_base_times)
            idle_s = time.time() - _last_event_ts

            if idle_s >= BUFFER_IDLE_CLEAR_S:
                print(f"[BUF] Cleared (hand off: streak={hand_off_count}, idle={idle_s:.2f}s)")
                letter_window.clear()

        # Extract ROI only if hand present
        if hand_present and rect is not None:
            roi = _crop_hand_roi_with_rect(frame, rect)
            if roi is not None and ENABLE_CLAHE_ROI:
                roi = _pre_enhance(roi)

        # Wrist tracking for wave (even if not present, but only add if landmarks)
        wrist_x = 0.5  # Default center
        curr_open_frac = 0.0
        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]
            wrist_x = lms.landmark[0].x  # Wrist landmark (0)
            curr_open_frac = _fingers_extended_fraction(lms)
        wave.add(wrist_x, curr_open_frac)

        # ---- Wave check (early, to veto others if active)
        now = time.time()
        if wave.is_wave() and not fired:
            if now - wave.last_spoken > wave.lockout_s and now - last_any_detection_at > GLOBAL_MIN_DETECTION_INTERVAL_S:
                word_wave = "hello" if wave.last_flips >= 5 else "hi"
                print(f"[WAVE@{frame_id}] '{word_wave}' (amp={wave.last_amp:.3f} flips={wave.last_flips} open={wave.last_open_frac:.2f})")
                _speak(word_wave)
                wave.last_spoken = now
                last_any_detection_at = now
                last_word_spoken_at = now
                letter_window.clear()
                fired = True
                word_hist[word_wave] += 1


        # Mute letters & patterns while wave lockout is active or motion is high
        wave_active = wave.is_active(now)
        high_motion = (wave.motion_level() >= wave.motion_std_veto)
        goto_inference = not (wave_active or high_motion)

        if WORD_DEBUG >= 2:
            if not goto_inference: print("[WORD-] skip: goto_inference=False")
            elif not ENABLE_WORD_DETECTION: print("[WORD-] skip: ENABLE_WORD_DETECTION=0")
            elif not WORD_PRED_OK: print("[WORD-] skip: WORD_PRED_OK=0")
            elif roi is None: print("[WORD-] skip: roi=None")
            elif (time.time() - last_word_ml_at) <= WORD_ML_COOLDOWN_S: print("[WORD-] skip: cooldown")
            elif score < WORD_MIN_SCORE: print(f"[WORD-] skip: score<{WORD_MIN_SCORE:.2f}")
            elif area  < WORD_MIN_AREA:  print(f"[WORD-] skip: area<{WORD_MIN_AREA:.2f}")

        # ---- Letter inference (throttled)
        if goto_inference and ENABLE_LETTER_DETECTION and hand_present and (frame_id % INFER_EVERY == 0) and roi is not None:
            letter, conf = predict_letter(roi)

            # Early floor filter
            EARLY_FLOOR = float(LETTER_THRESH.get("prob_threshold", 0.0))
            if conf < EARLY_FLOOR:
                total_R += 1
                curr_preds = [("low", conf)]
                continue  # skip dwell/buffer/pattern this frame

            curr_preds = [(letter, conf)]

            # Dwell tracking
            if letter == last_top: same_top_streak += 1
            else: same_top_streak = 1; last_top = letter

            thr = _thr_for(letter)

            thr_cap = 0.92 if letter not in ("J", "Z") else 0.97
            if letter in ("N", "O"):
                thr_cap = 0.85  # these were 0.995 in thresholds.json -> too strict live
            thr = min(thr, thr_cap)

            if letter == "unknown" or conf < thr:
                total_R += 1              # rejected/low-conf letter (count it)
            else:
                pred_hist[letter] += 1

                # --- N/O buffer veto during open-palm or high motion
                #skip_noisy = (
                    #BUFFER_VETO_NO
                    #and (letter in ("N", "O"))
                    #and (curr_open_frac >= 0.55 or wave.motion_level() >= (wave.motion_std_veto * 0.8))
                #)

                # Append to buffer only if allowed
                #if not skip_noisy and letter != "unknown":
                if letter != "unknown":
                    letter_window.append(letter)

                    # --- Pattern fallback (letters  words) uses the UPDATED buffer
                    if ENABLE_PATTERN_MATCHING and len(letter_window) >= 2 and not fired:
                        pat, word, comp = check_pattern(letter_window, WORD_PATTERNS, PATTERN_MATCH_LENGTH)
                        if pat and word:
                            now = time.time()
                            new_tail = comp[-len(pat):]

                            # avoid ON/NO while waving / recent wave / open-palm lateral motion
                            is_on_no = word in ("on", "no", "off")
                            recent_wave = wave.is_active(now) or (now - wave.last_spoken) < 1.0
                            open_or_motion = (curr_open_frac >= 0.55) or (wave.motion_level() >= wave.motion_std_veto)

                            if (now - last_word_spoken_at > WORD_COOLDOWN_S
                                and new_tail != last_compressed_tail
                                and now - last_any_detection_at > GLOBAL_MIN_DETECTION_INTERVAL_S
                                and not (is_on_no and (recent_wave or open_or_motion))):
                                print(f"[PATTERN@{frame_id}] '{pat}' '{word}' | comp='{comp}'")
                                _speak(word); total_P += 1
                                last_word_spoken_at = now
                                last_compressed_tail = new_tail
                                last_any_detection_at = now
                                letter_window.clear()
                                fired = True
                                word_hist[word] += 1

                    # Letter speak (dwell + majority)
                    if not fired and same_top_streak >= 3 and len(letter_window) >= LETTER_MAJORITY_WINDOW:
                        win = list(letter_window)[-LETTER_MAJORITY_WINDOW:]
                        common = max(set(win), key=win.count)
                        if win.count(common) >= int(0.7 * LETTER_MAJORITY_WINDOW) and common == letter:
                            now = time.time()
                            if now - last_letter_said_at[common] > LETTER_REPEAT_COOLDOWN_S \
                               and now - last_any_detection_at > GLOBAL_MIN_DETECTION_INTERVAL_S:

                                # motion gate keep audio clean if the hand is still moving
                                if wave.motion_level() <= LETTER_SPEAK_MAX_MOTION:
                                    print(f"[LETTER@{frame_id}] '{common}' conf={conf:.3f}")
                                    _speak(common); total_L += 1
                                    last_letter_said_at[common] = now
                                    last_any_detection_at = now
                                # else: optional debug
                                #     print(f"[LETTER] skipped speak due to motion={wave.motion_level():.3f} > {LETTER_SPEAK_MAX_MOTION:.3f}")

        # ---- Word-ML (robust, thresholded) -----------------------------------------
        if goto_inference and ENABLE_WORD_DETECTION and WORD_PRED_OK and roi is not None and hand_present and not fired:
            try:
                now = time.time()
                if (now - last_word_ml_at > WORD_ML_COOLDOWN_S
                    and score >= WORD_MIN_SCORE            # <-- tunable (env)
                    and area  >= WORD_MIN_AREA):           # <-- tunable (env)

                    w_label, w_conf = predict_word(roi)
                    if w_label:
                        w_norm = w_label.strip().lower()
                        wl_ok = (not WORDS_WHITELIST) or (w_norm in WORDS_WHITELIST)
                        need = _word_thr_for(w_norm)                       # per-word thr or env fallback
                        ok_motion = (wave.motion_level() <= LETTER_SPEAK_MAX_MOTION)

                        # Always show candidate when debugging (even if not whitelisted)
                        if WORD_DEBUG:
                            print(f"[WORD?] cand='{w_label}' conf={w_conf:.2f} need={need:.2f} "
                                  f"wl={'Y' if wl_ok else 'N'} score={score:.2f} area={area:.3f} "
                                  f"motion_ok={'Y' if ok_motion else 'N'}")

                        # Speak only when all gates pass
                        if (wl_ok and w_norm != 'unknown' and not w_norm.isdigit()
                            and w_conf >= need and ok_motion):
                            print(f"[WORD@{frame_id}]  '{w_label}' ({w_conf:.2f} {need:.2f})")
                            _speak(w_label)
                            last_word_ml_at = now
                            last_any_detection_at = now
                            fired = True   # prevent any other speak in this frame
                            word_hist[w_norm] += 1

            except Exception as e:
                print(f"[WORD][ERR] {e}")

        # GC
        if frame_id - last_gc >= GC_EVERY: gc.collect(); last_gc = frame_id

        # ---- VIS ----
        disp = frame.copy()
        if results.multi_hand_landmarks and DEBUG_VISUAL:
            for lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    disp, lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
                )
        cv2.putText(disp, "HAND" if hand_present else "NO-HAND", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0) if hand_present else (0,0,255), 2)
        buf_txt = ''.join(letter_window)[-15:] if letter_window else "(empty)"
        cv2.putText(disp, f"Buffer: {buf_txt}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(disp, f"score={score:.2f} area={area:.3f}  Conf>{CONF_THRESH:.2f}  INFER_EVERY={INFER_EVERY}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        if SHOW_CONFIDENCE_BARS: draw_conf_bars(disp, curr_preds, 10, 120)

        # HUD: show wave metrics (nice for tuning)
        cv2.putText(disp, f"wave_amp={wave.last_amp:.3f} flips={wave.last_flips} open={wave.last_open_frac:.2f} "
              f"motion={wave.motion_level():.3f} active={wave.is_active(time.time())}",
        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)

        cv2.imshow("SpeakEZ - Hybrid (MediaPipe + Models)", disp)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27): break
        if k == ord('s'):
            fn = f"screenshot_{frame_id}.png"; cv2.imwrite(fn, disp); print(f"[SYS] Saved: {fn}")

    cap.release(); cv2.destroyAllWindows()
    elapsed = time.time() - start
    fps = (frame_id/elapsed) if elapsed>0 else 0.0
    print("\n" + "="*70)
    print("SESSION SUMMARY")
    print("="*70)
    print(f"Duration: {elapsed:.1f}s | Frames: {frame_id} | FPS: {fps:.1f}")
    print(f"Letters: {total_L} | Patterns: {total_P} | Rejections: {total_R}")
    if pred_hist:
        total = sum(pred_hist.values())
        top, cnt = pred_hist.most_common(1)[0]
        print(f"Top letter '{top}' = {100*cnt/total:.1f}%")

    total_words = sum(word_hist.values())
    print(f"Words: {total_words} | Unique: {len(word_hist)}")
    if word_hist:
        for w, c in word_hist.most_common(10):
            print(f" {w}: {c}")

    if pred_hist:
        print("Letters (top 10):")
        for L, c in pred_hist.most_common(10):
            print(f" {L}: {c}")

if __name__ == "__main__":
    main()