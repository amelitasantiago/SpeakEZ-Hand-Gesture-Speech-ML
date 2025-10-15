# =============================
# ui/app_hybrid_simple.py (FIXED v1.3)
# =============================
# CRITICAL FIXES:
# 1. Face filtering in skin detection (lower ROI only)
# 2. Proper confidence distribution extraction
# 3. Fixed grace period reset logic
# 4. Added model confidence inspection
# 5. Emergency model bias detection

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import os, time, yaml, cv2, numpy as np, gc
from collections import deque, defaultdict, Counter

import tensorflow as tf
print(f"[SYS] TF {tf.__version__} | Eager: {tf.executing_eagerly()}")

# Load config
config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"[SYS] Loaded config.yaml")
else:
    config = {}

try:
    gc.collect()
    print(f"[SYS] ✓ GC module verified")
except Exception as e:
    print(f"[SYS][WARN] GC test failed: {e}")

# Path setup
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURR_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---- TTS ----
try:
    from tts import tts_speak_live as _speak
except Exception:
    def _speak(txt: str):
        print(f"[TTS] {txt}")

# ---- Letters ----
try:
    from inference_ import predict_letter
except Exception as e:
    print(f"[DET][ERR] inference_ import failed: {e}")
    predict_letter = None

# ---- Config ----
ENABLE_PATTERN_MATCHING = os.environ.get("ENABLE_PATTERN_MATCHING", "true").lower() == "true"

# STRICT thresholds to combat model bias
CONF_THRESH = float(os.environ.get("CONF_THRESH", 0.90))  # Raised to 0.90
MIN_CONFIDENCE_MARGIN = float(os.environ.get("MIN_CONFIDENCE_MARGIN", 0.30))  # Raised to 0.30
LETTER_MAJORITY_WINDOW = int(os.environ.get("LETTER_MAJORITY_WINDOW", 7))  # More samples
LETTER_REPEAT_COOLDOWN_S = float(os.environ.get("LETTER_REPEAT_COOLDOWN_S", 1.5))
BUFFER_GRACE_PERIOD_FRAMES = int(os.environ.get("BUFFER_GRACE_PERIOD_FRAMES", 8))

# Hand detection - STRICTER to avoid face
HAND_MOTION_THR = float(os.environ.get("HAND_MOTION_THR", 0.025))  # Raised
HAND_SKIN_THR = float(os.environ.get("HAND_SKIN_THR", 0.12))  # Raised significantly
HAND_WARMUP = int(os.environ.get("HAND_WARMUP", 30))
HAND_MIN_FRAMES = int(os.environ.get("HAND_MIN_FRAMES", 7))  # More consecutive frames

GLOBAL_MIN_DETECTION_INTERVAL_S = 0.8

# Pattern config
PATTERN_MATCH_LENGTH = 10
WORD_PATTERNS = {
    "NO": "no", "NOO": "no", "NOOO": "no",
    "ON": "on", "OFF": "off", "OF": "off",
    "HELLO": "hello", "HI": "hi", "HELP": "help",
    "OK": "okay", "YES": "yes",
}

# Camera
CAM_INDEX = 0
TARGET_W, TARGET_H = 640, 480
TARGET_FPS = 30

# Debug
DEBUG_VISUAL = os.environ.get("DEBUG_VISUAL", "true").lower() == "true"
DEBUG_PREDICTIONS = os.environ.get("DEBUG_PREDICTIONS", "true").lower() == "true"
SHOW_CONFIDENCE_BARS = True
LOG_EVERY_S = 0.5

# ---- Enhanced skin detection with FACE EXCLUSION ----
def _skin_mask_ycrcb_lower_only(frame_bgr):
    """
    Skin detection EXCLUDING upper region (where face typically is)
    Only look in LOWER 60% of frame to avoid face detection
    """
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    
    h, w = mask.shape
    
    # CRITICAL: Only look in LOWER portion where hands typically are
    # Exclude top 40% to filter out face
    face_cutoff = int(h * 0.40)  # Everything above this is zeroed out
    mask[:face_cutoff, :] = 0
    
    # Further restrict to center horizontal region
    roi_x1, roi_x2 = int(w * 0.15), int(w * 0.85)
    roi_mask = np.zeros_like(mask)
    roi_mask[face_cutoff:, roi_x1:roi_x2] = mask[face_cutoff:, roi_x1:roi_x2]
    
    return roi_mask

def _ratio(mask):
    return np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])

def _motion_ratio(gray, prev_gray):
    if prev_gray is None:
        return 0.0
    diff = cv2.absdiff(gray, prev_gray)
    return np.mean(diff) / 255.0

def _open_cam():
    print(f"[CAM] Opening camera {CAM_INDEX}")
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"[CAM][FATAL] Cannot open camera {CAM_INDEX}")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    
    ret, test = cap.read()
    if not ret:
        raise RuntimeError("[CAM][FATAL] Cannot read frames")
    
    print(f"[CAM] ✓ Opened: {test.shape}")
    return cap

def check_pattern_match(letter_buffer, patterns, lookback=10):
    if not letter_buffer or len(letter_buffer) < 2:
        return None, None
    
    recent = ''.join(list(letter_buffer)[-lookback:])
    sorted_patterns = sorted(patterns.items(), key=lambda x: len(x[0]), reverse=True)
    
    for pattern, word in sorted_patterns:
        if pattern in recent:
            return pattern, word
    
    return None, None

def draw_confidence_bars(frame, predictions, x=10, y=120):
    """Draw confidence bars for predictions"""
    if not predictions or len(predictions) == 0:
        cv2.putText(frame, "No predictions", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return
    
    bar_width = 180
    bar_height = 18
    spacing = 23
    
    for i, (letter, conf) in enumerate(predictions[:5]):  # Show top 5
        y_pos = y + (i * spacing)
        
        # Background
        cv2.rectangle(frame, (x, y_pos), (x + bar_width, y_pos + bar_height),
                     (40, 40, 40), -1)
        
        # Confidence bar
        fill_width = int(bar_width * conf)
        if i == 0:
            color = (0, 255, 0) if conf >= CONF_THRESH else (0, 165, 255)
        else:
            color = (100, 100, 100)
        
        cv2.rectangle(frame, (x, y_pos), (x + fill_width, y_pos + bar_height),
                     color, -1)
        
        # Text
        text = f"{letter}: {conf:.3f}"
        cv2.putText(frame, text, (x + bar_width + 10, y_pos + 14),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

# ---- Get confidence distribution from model ----
def get_confidence_distribution(frame):
    """
    Extract FULL confidence distribution from model
    Returns: [(letter, confidence), ...] sorted by confidence
    """
    try:
        # Try basic prediction first
        result = predict_letter(frame)
        
        # If it returns just (letter, conf), we need to inspect the model directly
        if isinstance(result, tuple) and len(result) == 2:
            letter, conf = result
            
            # Try to get the model object to extract full distribution
            try:
                from inference_ import letter_model, LETTER_LABELS
                if letter_model is not None:
                    # Extract features (mimicking predict_letter internals)
                    from inference_ import extract_letter_features
                    feats = extract_letter_features(frame)
                    
                    if feats is not None:
                        feats = np.asarray(feats).reshape(1, -1)
                        
                        # Get probabilities
                        if hasattr(letter_model, 'predict_proba'):
                            proba = letter_model.predict_proba(feats)[0]
                            
                            # Build distribution
                            distribution = []
                            for i, label in enumerate(LETTER_LABELS):
                                distribution.append((str(label), float(proba[i])))
                            
                            # Sort by confidence
                            distribution.sort(key=lambda x: x[1], reverse=True)
                            return distribution
            except Exception as e:
                if DEBUG_PREDICTIONS:
                    print(f"[CONF][WARN] Could not extract distribution: {e}")
            
            # Fallback: just return single prediction
            return [(letter, conf)]
        
        # If it already returns distribution
        elif isinstance(result, tuple) and len(result) == 3:
            return result[2]  # (letter, conf, distribution)
        
        return []
        
    except Exception as e:
        print(f"[CONF][ERR] {e}")
        return []

def main():
    cap = None
    try:
        cap = _open_cam()
    except RuntimeError as e:
        print(f"[SYS][FATAL] {e}")
        return

    # State
    letter_window = deque(maxlen=max(LETTER_MAJORITY_WINDOW, PATTERN_MATCH_LENGTH))
    last_letter_said_at = defaultdict(float)
    last_pattern_said_at = defaultdict(float)
    last_any_detection_at = 0.0
    prev_gray = None
    frame_id = 0
    last_log_frame = 0
    start_time = time.time()
    
    # Hand tracking - FIXED grace period
    hand_present_frames = 0
    frames_since_hand_lost = 0
    buffer_was_active_last_frame = False
    
    # Statistics
    total_letter_detections = 0
    total_pattern_detections = 0
    total_rejections = 0
    rejection_reasons = Counter()
    letter_detection_history = []
    pattern_detection_history = []
    prediction_diversity = Counter()
    
    last_gc_frame = 0
    GC_EVERY_N_FRAMES = 300

    print(f"[SYS] Starting main loop (press 'q' or ESC to exit)")
    print(f"[SYS] STRICT ANTI-BIAS MODE:")
    print(f"[SYS]   - Confidence threshold: {CONF_THRESH:.2f}")
    print(f"[SYS]   - Margin requirement: {MIN_CONFIDENCE_MARGIN:.2f}")
    print(f"[SYS]   - Majority window: {LETTER_MAJORITY_WINDOW} frames")
    print(f"[SYS]   - Hand skin threshold: {HAND_SKIN_THR:.2f} (FACE EXCLUDED)")
    print(f"[SYS]   - Hand motion threshold: {HAND_MOTION_THR:.3f}")
    print(f"[SYS] Pattern matching: {'ENABLED' if ENABLE_PATTERN_MATCHING else 'DISABLED'}")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_id += 1

        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.std(gray) < 2.5:
            continue

        # Hand detection with face exclusion
        skin_mask = _skin_mask_ycrcb_lower_only(frame)
        skin_ratio = _ratio(skin_mask)
        motion_ratio = _motion_ratio(gray, prev_gray)
        adj_motion = motion_ratio + (skin_ratio * 0.015)  # Reduced boost
        
        hand_detected_raw = (
            (adj_motion > HAND_MOTION_THR) and 
            (skin_ratio > HAND_SKIN_THR) and 
            (frame_id > HAND_WARMUP)
        )
        
        if hand_detected_raw:
            hand_present_frames += 1
            frames_since_hand_lost = 0  # RESET when hand detected
        else:
            hand_present_frames = 0
            frames_since_hand_lost += 1  # INCREMENT when no hand
        
        hand_present = hand_present_frames >= HAND_MIN_FRAMES
        
        # Buffer grace logic - FIXED
        buffer_active = hand_present or (frames_since_hand_lost < BUFFER_GRACE_PERIOD_FRAMES)
        
        # Clear buffer when grace period expires
        if buffer_was_active_last_frame and not buffer_active:
            if len(letter_window) > 0:
                if DEBUG_PREDICTIONS:
                    print(f"[BUF@{frame_id}] Grace expired ({frames_since_hand_lost}f) - clearing buffer")
                letter_window.clear()
        
        buffer_was_active_last_frame = buffer_active

        # Logging
        if frame_id - last_log_frame >= int(LOG_EVERY_S * TARGET_FPS):
            fps = frame_id / (time.time() - start_time)
            buf_str = ''.join(letter_window)[-15:] if letter_window else ''
            hand_str = f"Hand:{'YES' if hand_present else 'NO'}({hand_present_frames}f)"
            grace_str = f"Grace:{frames_since_hand_lost}/{BUFFER_GRACE_PERIOD_FRAMES}" if not hand_present else ""
            stats_str = f"L:{total_letter_detections} P:{total_pattern_detections} R:{total_rejections}"
            print(f"F{frame_id}: FPS={fps:.1f} | {hand_str} {grace_str} | m={adj_motion:.3f} s={skin_ratio:.3f} | Buf:'{buf_str}' | {stats_str}")
            last_log_frame = frame_id

        fired = False
        current_predictions = []
        
        # ----- LETTER DETECTION -----
        if buffer_active and predict_letter is not None:
            try:
                # Get FULL confidence distribution
                distribution = get_confidence_distribution(frame)
                current_predictions = distribution  # For visualization
                
                if not distribution or len(distribution) == 0:
                    continue
                
                # Top prediction
                letter, conf = distribution[0]
                
                # Track diversity
                if letter and letter != "unknown":
                    prediction_diversity[letter] += 1
                
                # Log predictions periodically
                if DEBUG_PREDICTIONS and frame_id % 15 == 0:
                    top_5 = distribution[:5]
                    pred_str = ", ".join([f"{l}:{c:.3f}" for l, c in top_5])
                    print(f"[PRED@{frame_id}] {pred_str}")
                
                # Confidence margin check
                second_conf = distribution[1][1] if len(distribution) > 1 else 0.0
                confidence_margin = conf - second_conf
                
                # STRICT FILTERING
                reject_reason = None
                if letter == "unknown":
                    reject_reason = "unknown"
                elif conf < CONF_THRESH:
                    reject_reason = f"low_conf({conf:.2f}<{CONF_THRESH})"
                elif confidence_margin < MIN_CONFIDENCE_MARGIN:
                    reject_reason = f"low_margin({confidence_margin:.2f}<{MIN_CONFIDENCE_MARGIN})"
                
                if reject_reason:
                    total_rejections += 1
                    rejection_reasons[reject_reason] += 1
                    
                    if DEBUG_PREDICTIONS and frame_id % 30 == 0:
                        second_letter = distribution[1][0] if len(distribution) > 1 else "?"
                        print(f"[REJECT@{frame_id}] '{letter}' rejected: {reject_reason} | 2nd='{second_letter}' ({second_conf:.3f})")
                    continue
                
                # PASSED all checks - add to buffer
                letter_window.append(letter)
                
                # Pattern matching
                if ENABLE_PATTERN_MATCHING and len(letter_window) >= 2:
                    pattern, word = check_pattern_match(letter_window, WORD_PATTERNS, PATTERN_MATCH_LENGTH)
                    
                    if pattern and word:
                        now = time.time()
                        if (buffer_active and
                            now - last_pattern_said_at[pattern] > 2.0 and
                            now - last_any_detection_at >= GLOBAL_MIN_DETECTION_INTERVAL_S):
                            
                            print(f"[PATTERN@{frame_id}] ✅ '{pattern}' → '{word}' | Buffer: {''.join(list(letter_window)[-20:])}")
                            _speak(word)
                            last_pattern_said_at[pattern] = now
                            last_any_detection_at = now
                            total_pattern_detections += 1
                            pattern_detection_history.append((pattern, word))
                            letter_window.clear()
                            fired = True
                
                # Individual letter speech
                if not fired and len(letter_window) >= LETTER_MAJORITY_WINDOW:
                    win = list(letter_window)[-LETTER_MAJORITY_WINDOW:]
                    common = max(set(win), key=win.count)
                    win_count = win.count(common)
                    win_pct = win_count / len(win)
                    
                    # Require strong majority (>60%)
                    if common == letter and win_pct > 0.6:
                        now = time.time()
                        if (now - last_letter_said_at[common] > LETTER_REPEAT_COOLDOWN_S and
                            now - last_any_detection_at >= GLOBAL_MIN_DETECTION_INTERVAL_S):
                            
                            print(f"[LETTER@{frame_id}] ✅ '{common}' conf={conf:.3f} margin={confidence_margin:.3f} win={win_count}/{len(win)} → SPEAK")
                            _speak(common)
                            last_letter_said_at[common] = now
                            last_any_detection_at = now
                            total_letter_detections += 1
                            letter_detection_history.append(common)
                        
            except Exception as e:
                print(f"[DET][ERR] {e}")

        prev_gray = gray.copy()

        # GC
        if frame_id - last_gc_frame >= GC_EVERY_N_FRAMES:
            gc.collect()
            last_gc_frame = frame_id

        # ----- VISUALIZATION -----
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Hand detection ROI (lower region only)
        if DEBUG_VISUAL:
            face_cutoff = int(h * 0.40)
            roi_x1, roi_x2 = int(w * 0.15), int(w * 0.85)
            
            # Draw ROI rectangle
            cv2.rectangle(display, (roi_x1, face_cutoff), (roi_x2, h-10), (0, 255, 255), 2)
            cv2.putText(display, "HAND ROI (Face Excluded)", (roi_x1+5, face_cutoff+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Skin overlay (only in ROI)
            skin_overlay = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
            skin_overlay[:, :, 1] = skin_mask
            display = cv2.addWeighted(display, 0.7, skin_overlay, 0.3, 0)
        
        # Status
        status_color = (0, 255, 0) if hand_present else (128, 128, 128) if buffer_active else (0, 0, 255)
        status_text = "ACTIVE" if hand_present else f"GRACE({frames_since_hand_lost})" if buffer_active else "IDLE"
        cv2.putText(display, f"Hand: {status_text}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Buffer
        buf_display = ''.join(list(letter_window)[-15:]) if letter_window else "(empty)"
        cv2.putText(display, f"Buffer: {buf_display}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Thresholds display
        cv2.putText(display, f"Conf>{CONF_THRESH:.2f} Margin>{MIN_CONFIDENCE_MARGIN:.2f}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Confidence bars
        if SHOW_CONFIDENCE_BARS:
            draw_confidence_bars(display, current_predictions, x=10, y=105)
        
        # Stats
        cv2.putText(display, f"L:{total_letter_detections} P:{total_pattern_detections} Reject:{total_rejections}",
                   (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("SpeakEZ - Diagnostic v1.3", display)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # ----- SUMMARY -----
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"SESSION SUMMARY")
    print(f"{'='*70}")
    print(f"Duration: {elapsed:.1f}s | Frames: {frame_id} | FPS: {frame_id/elapsed:.1f}")
    print(f"\nDetections:")
    print(f"  Letters: {total_letter_detections}")
    print(f"  Patterns: {total_pattern_detections}")
    print(f"  Rejections: {total_rejections}")
    
    if rejection_reasons:
        print(f"\nRejection Breakdown:")
        for reason, count in rejection_reasons.most_common():
            print(f"  {reason}: {count}")
    
    if letter_detection_history:
        letter_counts = Counter(letter_detection_history)
        print(f"\nLetter Detections:")
        for letter, count in letter_counts.most_common():
            print(f"  '{letter}': {count}")
    
    if pattern_detection_history:
        print(f"\n✅ Patterns Matched:")
        for pattern, word in pattern_detection_history:
            print(f"  '{pattern}' → '{word}'")
    else:
        print(f"\nℹ️  No patterns matched")
    
    # Diversity analysis
    if prediction_diversity:
        total_preds = sum(prediction_diversity.values())
        print(f"\n📊 Prediction Diversity:")
        print(f"  Total predictions: {total_preds}")
        print(f"  Unique letters: {len(prediction_diversity)}")
        print(f"  Top predictions:")
        for letter, count in prediction_diversity.most_common(10):
            pct = (count / total_preds) * 100
            bar = '█' * int(pct / 5)
            print(f"    '{letter}': {count:4d} ({pct:5.1f}%) {bar}")
        
        # Bias warning
        if len(prediction_diversity) > 0:
            top_letter, top_count = prediction_diversity.most_common(1)[0]
            top_pct = (top_count / total_preds) * 100
            if top_pct > 80:
                print(f"\n⚠️  MODEL BIAS DETECTED!")
                print(f"   '{top_letter}' represents {top_pct:.1f}% of predictions")
                print(f"   RECOMMENDED ACTIONS:")
                print(f"   1. Retrain model with balanced dataset")
                print(f"   2. Check training data distribution")
                print(f"   3. Verify feature extraction quality")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[SYS] Interrupted")
        try:
            from tts import _get_speaker
            _get_speaker().close(timeout=2.0)
        except:
            pass
        cv2.destroyAllWindows()
        print("[SYS] Exit")