# C:\Users\ameli\speakez\src\inference_.py (GR)
import os, json, cv2, numpy as np
import tensorflow as tf
from pathlib import Path
from collections import deque

# --- CONFIG ---
_IMG_SIZE = 128
_PROJ = Path(__file__).resolve().parents[1]  # repo root (one level above src/)
_MODEL_PATH = os.getenv(
    "SPEAKEZ_LETTERS_MODEL",
    str(_PROJ / "models" / "final" / "asl_baseline_cnn_128_final.h5"),
)
_CLASSES_JSON = os.getenv(
    "SPEAKEZ_LETTERS_CLASSES",
    str(_PROJ / "models" / "final" / "classes_letters.json"),
)

# Load word whitelist independently
ws_path = _PROJ / "models" / "final" / "words_subset.json"
if ws_path.exists():
    with open(ws_path, "r") as f:
        _ws = json.load(f)
    if isinstance(_ws, dict):
        words_list = _ws.get("words", [])
    else:
        words_list = _ws
    WORDS_WHITELIST = {str(w).strip().lower() for w in words_list if isinstance(w, (str, int))}
else:
    WORDS_WHITELIST = set()  # Fallback: allow all if missing
    print("[SYS] ⚠ No words_subset.json; allowing all word labels")


# --- LOAD MODEL & CLASSES ---
_letters_model = tf.keras.models.load_model(_MODEL_PATH, compile=False)

if os.path.exists(_CLASSES_JSON):
    with open(_CLASSES_JSON, "r") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        LETTER_CLASSES = [raw[str(i)] for i in range(len(raw))]
    else:
        LETTER_CLASSES = raw
else:
    LETTER_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["DEL", "SPACE", "NOTHING"]

# --- SIMPLE CENTER CROP (no mediapipe dependency) ---
def _center_square_crop(img_bgr):
    h, w = img_bgr.shape[:2]
    side = min(h, w)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return img_bgr[y0:y0 + side, x0:x0 + side]

def _preprocess_bgr(img_bgr):
    crop = _center_square_crop(img_bgr)
    img = cv2.resize(crop, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # match training color order
    img = img.astype(np.float32) / 255.0        # match training normalization
    return img

# Decorate the core inference for graph-mode stability
@tf.function  # Compiles to graph; reduces eager leaks (TF 2.0+ compatible)
def _stable_model_call(model, inputs):
    return model(inputs, training=False)

def predict_letter(img_bgr, conf_threshold=0.30):
    """
    Input: BGR frame (np.uint8 HxWx3). Output: (token, confidence)
    token in LETTER_CLASSES or 'unknown' if below threshold.
    """
    x = _preprocess_bgr(img_bgr)[None, ...]  # (1,128,128,3)
    try:
        probs = _stable_model_call(_letters_model, x).numpy()[0]  # Graph-pinned call
    except Exception as e:
        print(f"[DET][TF-ERR] Model call failed: {e}")
        return "unknown", 0.0
    top = int(np.argmax(probs))
    conf = float(probs[top])
    label = LETTER_CLASSES[top] if top < len(LETTER_CLASSES) else "unknown"
    if conf < conf_threshold:
        return "unknown", conf
    return label, conf

# Compatibility alias
def detect_letter(img_bgr, model=None):
    return predict_letter(img_bgr)

# ========== WORD FEATURE EXTRACTOR (170-dim) ==========
# Matches your LogisticRegression pipeline expectation.
def extract_word_features(frame_bgr):
    """
    Return np.float32 vector of length 170 expected by word_skel_logreg_8.joblib.
    Also returns skin_frac for gating.
    Composition (CPU-cheap, deterministic):
      7  = Hu moments (log-scaled) on YCrCb skin mask
      6  = contour geometry stats (area, extent, solidity, aspect, vx, ecc)
      80 = spatial gray hist (global + 2x2, 16 bins each)
      48 = Y/Cr/Cb global hists (16 bins ×3)
      16 = Gabor energies (4 scales × 4 orientations)
      13 = Sobel magnitude histogram (13 bins)
      Total = 170 dims
    """
    def _skin_mask_ycrcb(bgr):
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, (0,133,77), (255,173,127))
        return cv2.medianBlur(mask, 5)

    def _hu(mask):
        m = cv2.moments(mask)
        hu = cv2.HuMoments(m).flatten()
        return (np.sign(hu) * np.log1p(np.abs(hu))).astype(np.float32)

    def _shape(mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return np.zeros(6, np.float32)
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        extent = area / (w*h + 1e-6)
        hull = cv2.convexHull(c)
        solidity = area / (cv2.contourArea(hull) + 1e-6)
        aspect = w / (h + 1e-6)
        if len(c) >= 2:
            vx,vy,x0,y0 = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        else:
            vx = 1.0
        pts = c.reshape(-1,2).astype(np.float32)
        cov = np.cov(pts.T)
        if cov.shape == (2,2):
            evals, _ = np.linalg.eig(cov)
            s1, s2 = np.sort(np.maximum(evals, 1e-6))[::-1]
            ecc = float(np.sqrt(1 - (s2/s1)))
        else:
            ecc = 0.0
        return np.array([area, extent, solidity, aspect, float(vx), float(ecc)], np.float32)

    def _spatial_gray(gray, bins=16):
        h, w = gray.shape
        h2, w2 = h//2, w//2
        parts = [gray, gray[:h2,:w2], gray[:h2,w2:], gray[h2:,:w2], gray[h2:,w2:]]
        out = []
        for p in parts:
            hist = cv2.calcHist([p],[0],None,[bins],[0,256]).flatten().astype(np.float32)
            hist /= (hist.sum() + 1e-6)
            out.append(hist)
        return np.concatenate(out)

    def _ycrcb_hists(img):
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        out = []
        for ch in range(3):
            h = cv2.calcHist([ycrcb],[ch],None,[16],[0,256]).flatten().astype(np.float32)
            h /= (h.sum() + 1e-6)
            out.append(h)
        return np.concatenate(out)

    def _gabor_energies(gray):
        out = []
        ks = [7,11,15,19]  # 4 scales
        thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 4 orientations
        for k in ks:
            for t in thetas:
                kern = cv2.getGaborKernel((k,k), sigma=0.56*k, theta=t, lambd=0.5*k, gamma=0.5, psi=0)
                f = cv2.filter2D(gray, cv2.CV_32F, kern)
                out.append(float(np.mean(np.abs(f))))
        return np.array(out, np.float32)  # 16

    def _sobel_hist(gray):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1,0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0,1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        hist = cv2.calcHist([mag.astype(np.uint8)],[0],None,[13],[0,256]).flatten().astype(np.float32)
        hist /= (hist.sum() + 1e-6)
        return hist  # 13

    # center square crop (safe fallback if you don't have a hand-cropper)
    H,W = frame_bgr.shape[:2]
    side = min(H,W)
    x0 = (W - side)//2; y0 = (H - side)//2
    crop = frame_bgr[y0:y0+side, x0:x0+side]
    img = cv2.resize(crop, (128,128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = _skin_mask_ycrcb(img)

    # Compute skin fraction for gating
    skin_frac = cv2.countNonZero(mask) / mask.size

    feats = np.concatenate([
        _hu(mask),                  # 7
        _shape(mask),               # 6
        _spatial_gray(gray, 16),    # 80
        _ycrcb_hists(img),          # 48
        _gabor_energies(gray),      # 16
        _sobel_hist(gray),          # 13
    ]).astype(np.float32)

    if feats.shape[0] != 170:
        z = np.zeros(170, np.float32)
        z[:min(170, feats.shape[0])] = feats[:min(170, feats.shape[0])]
        feats = z
    return feats, skin_frac

# ========== Fix Word Prediction (Enable predict_word) ==========
from joblib import load
from sklearn.exceptions import NotFittedError  # For error handling

# Word model and classes (match words_subset.json)
_WORD_MODEL_PATH = str(_PROJ / "models" / "final" / "word_skel_logreg_8.joblib")
try:
    _words_model = load(_WORD_MODEL_PATH)
except Exception as e:
    raise ImportError(f"Failed to load word model: {e}")

with open(str(_PROJ / "models" / "final" / "words_subset.json"), "r") as f:
    _ws = json.load(f)
    WORD_CLASSES = _ws.get("words", [])  # ["yes", "no", "please", ...]

def predict_word(img_bgr, conf_threshold=0.25):  # Default from config/thresholds.json
    """
    Input: BGR ROI (np.uint8 HxWx3). Output: (label, confidence) or (None, 0.0) if invalid.
    Uses logistic regression on 170-dim features. Filters by whitelist and threshold.
    """
    try:
        feats, skin_frac = extract_word_features(img_bgr)
        if skin_frac < float(os.environ.get("MIN_SKIN_FRAC", "0.15")):
            return None, 0.0  # Insufficient skin; likely no hand or background
        feats = feats.reshape(1, -1)  # (1, 170)
        probs = _words_model.predict_proba(feats)[0]  # Softmax-like probs
        top_idx = np.argmax(probs)
        conf = float(probs[top_idx])
        label = WORD_CLASSES[top_idx] if top_idx < len(WORD_CLASSES) else "unknown"

        if conf < conf_threshold or label.lower() not in WORDS_WHITELIST:
            return None, 0.0
        return label, conf
    except NotFittedError:
        print("[WORD][ERR] Model not fitted.")
        return None, 0.0
    except Exception as e:
        print(f"[WORD][ERR] Prediction failed: {type(e).__name__} - {str(e)}")
        return None, 0.0