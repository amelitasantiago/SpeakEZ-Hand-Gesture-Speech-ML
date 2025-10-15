import numpy as np, cv2
from src.inference import predict_letter

def test_predict_letter_smoke():
    dummy = np.zeros((256,256,3), np.uint8)  # black frame
    tok, conf = predict_letter(dummy)
    assert isinstance(tok, str)
    assert 0.0 <= conf <= 1.0
