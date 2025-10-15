import os, cv2, numpy as np
from typing import Tuple
from utils import IDX2TOK, resolve, load_config

# Lazy TF import to keep CLI tools fast
import tensorflow as tf

def _load_letters_model():
    cfg = load_config()
    kpath = resolve(cfg["paths"]["letters_model_keras"])
    hpath = resolve(cfg["paths"]["letters_model_h5"])
    model = None
    if os.path.exists(kpath):
        model = tf.keras.models.load_model(kpath, compile=False)
        which = kpath
    elif os.path.exists(hpath):
        model = tf.keras.models.load_model(hpath, compile=False)
        which = hpath
    else:
        raise FileNotFoundError("Letters model not found (.keras or .h5) under models/final/")
    return model, which

LETTERS_MODEL, LETTERS_MODEL_PATH = _load_letters_model()

def preprocess_letter_frame(bgr, size=(128,128)):
    h,w = bgr.shape[:2]; m=min(h,w); y0=(h-m)//2; x0=(w-m)//2
    crop = bgr[y0:y0+m, x0:x0+m]
    img  = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    return img

def predict_letter(bgr) -> Tuple[str, float]:
    x = preprocess_letter_frame(bgr)
    y = LETTERS_MODEL.predict(x[None,...], verbose=0)[0]
    idx = int(np.argmax(y)); tok = IDX2TOK[idx]; conf = float(y[idx])
    return tok, conf

class LetterBuffer:
    """Majority smoothing + editable text buffer."""
    def __init__(self, N=7):
        from collections import deque
        self.deq = deque(maxlen=N)
        self.last = None
        self.buf  = ""

    def push(self, tok:str):
        if tok != "NOTHING":
            self.deq.append(tok)
        if len(self.deq) < 4:
            return None
        vals, counts = np.unique(self.deq, return_counts=True)
        winner = vals[counts.argmax()]
        if winner != self.last:
            self.last = winner
            self._apply(winner)
            return winner
        return None

    def _apply(self, tok):
        if tok == "DEL":
            self.buf = self.buf[:-1]
        elif tok == "SPACE":
            self.buf += " "
        else:
            self.buf += tok
