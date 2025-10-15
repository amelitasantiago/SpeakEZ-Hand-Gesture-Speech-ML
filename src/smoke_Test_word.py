import json, joblib, numpy as np, os
from tensorflow.keras.models import load_model

ROOT = "."
def assert_exists(p): 
    if not os.path.exists(p): raise FileNotFoundError(p)

letters_h5 = "models/final/asl_baseline_cnn_128_final.h5"
letters_keras = "models/final/best_asl_baseline_128.keras"
letters_model = letters_h5 if os.path.exists(letters_h5) else letters_keras
assert_exists(letters_model)
m = load_model(letters_model, compile=False)
print("Letters model OK:", letters_model)

logreg = "models/final/word_skel_logreg_8.joblib"
assert_exists(logreg)
clf = joblib.load(logreg)
print("Words model OK:", logreg, "→ classes:", getattr(clf, "classes_", None))

th = json.load(open("models/final/thresholds.json"))
print("Thresholds:", th)

ws = json.load(open("models/final/words_subset.json"))
print("Words subset:", ws)

proto = "models/final/word_skel_prototypes_8.npz"
print("Prototypes:", os.path.exists(proto))
