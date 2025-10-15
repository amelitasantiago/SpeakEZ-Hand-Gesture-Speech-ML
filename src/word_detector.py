import os, json, cv2, numpy as np, joblib
from typing import Dict, Tuple
from utils import resolve, load_config

import mediapipe as mp
mp_hands = mp.solutions.hands

def _load_words_artifacts():
    cfg = load_config()
    lr_path  = resolve(cfg["paths"]["word_logreg"])
    thr_path = resolve(cfg["paths"]["word_thresholds"])
    sub_path = resolve(cfg["paths"]["word_subset"])
    proto_p  = resolve(cfg["paths"]["word_prototypes"])

    if not os.path.exists(lr_path):
        raise FileNotFoundError("word_skel_logreg_8.joblib not found.")
    lr = joblib.load(lr_path)

    thr_cfg = {"prob_threshold": cfg["ui"]["prob_threshold_default"],
               "top2_margin": cfg["ui"]["top2_margin_default"],
               "agreement": cfg["ui"]["agreement_default"]}
    if os.path.exists(thr_path):
        thr_cfg.update(json.load(open(thr_path)))

    words = json.load(open(sub_path))
    if isinstance(words, dict) and "words" in words: words = words["words"]
    assert isinstance(words, list) and len(words) == len(lr.classes_), "words_subset mismatch classes"

    protos = None
    if os.path.exists(proto_p):
        try:
            protos = np.load(proto_p)["protos"]
        except Exception:
            protos = None
    return lr, words, thr_cfg, protos

LR, WORDS, THR_CFG, PROTOS = _load_words_artifacts()

def read_video_uniform_rgb(path, T=24, size=(128,128)):
    cap = cv2.VideoCapture(path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(frames-1,0), T).astype(int) if frames>0 else np.zeros(T, int)
    out = np.zeros((T, size[1], size[0], 3), np.float32)
    f=0; ok, frame = cap.read()
    while ok:
        if f in idxs:
            h,w = frame.shape[:2]; m=min(h,w); y0=(h-m)//2; x0=(w-m)//2
            crop = frame[y0:y0+m, x0:x0+m]; crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
            rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            pos  = np.where(idxs==f)[0][0]; out[pos] = rgb
        ok, frame = cap.read(); f+=1
    cap.release(); return out

def normalize_hand_landmarks(hand_xyz):
    xyz = hand_xyz.copy()
    wrist = xyz[0]; xyz -= wrist
    ref = (np.linalg.norm(xyz[9]) + np.linalg.norm(xyz[13]))/2.0 + 1e-6
    xyz /= ref
    return xyz.astype(np.float32)

def extract_seq_hands(frames_rgb01):
    seq = []
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.4) as hands:
        for t in range(frames_rgb01.shape[0]):
            img = (frames_rgb01[t]*255).astype(np.uint8)
            res = hands.process(img)
            hands_norm=[]
            if res.multi_hand_landmarks:
                for lm in res.multi_hand_landmarks:
                    arr = np.array([[p.x,p.y,p.z] for p in lm.landmark], dtype=np.float32)
                    hands_norm.append(arr)
                hands_norm.sort(key=lambda a: a[0,0])
            while len(hands_norm) < 2: hands_norm.append(None)
            H0 = normalize_hand_landmarks(hands_norm[0]) if hands_norm[0] is not None else None
            H1 = normalize_hand_landmarks(hands_norm[1]) if hands_norm[1] is not None else None
            seq.append((H0,H1))
    return seq

def clip_features_from_sequence(seq):
    feats=[]
    for hand in [0,1]:
        traj=[]
        for (H0,H1) in seq:
            H = H0 if hand==0 else H1
            traj.append(np.zeros((21,3), np.float32) if H is None else H)
        X = np.stack(traj, axis=0)
        V = np.diff(X, axis=0, prepend=X[:1])
        f = np.concatenate([X.mean((0,2)), X.std((0,2)), V.mean((0,2)), V.std((0,2))], axis=0)
        feats.append(f)
    feat = np.concatenate(feats, axis=0).astype(np.float32)
    if feat.shape[0] < 170: feat = np.pad(feat, (0, 170-feat.shape[0]))
    feat = feat[:170]
    return feat / (np.linalg.norm(feat)+1e-9)

def classify_word_video(video_path:str,
                        thr:float=None, margin:float=None, agreement:bool=None):
    if thr is None: thr = float(THR_CFG.get("prob_threshold", 0.25))
    if margin is None: margin = float(THR_CFG.get("top2_margin", 0.0))
    if agreement is None: agreement = bool(THR_CFG.get("agreement", False))

    clip = read_video_uniform_rgb(video_path, T=24, size=(128,128))
    seq  = extract_seq_hands(clip)
    feat = clip_features_from_sequence(seq)
    proba = LR.predict_proba(feat[None,:])[0]
    order = np.argsort(proba); p1i, p2i = int(order[-1]), int(order[-2])
    p1, p2 = float(proba[p1i]), float(proba[p2i])
    accept = (p1>=thr) and ((p1-p2)>=margin)

    if agreement and PROTOS is not None:
        # cosine nearest prototype must match
        nn = int(np.argmax(feat[None,:] @ PROTOS.T))
        accept = accept and (nn == p1i)

    label = WORDS[p1i] if accept else "unknown"
    probs = {WORDS[i]: float(proba[i]) for i in range(len(WORDS))}
    return label, p1, probs
