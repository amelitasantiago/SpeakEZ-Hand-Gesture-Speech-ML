import cv2, numpy as np
from inference_ import predict_letter  # adjust import to match your layout

dummy = np.zeros((480,640,3), dtype=np.uint8)  # black frame
print(predict_letter(dummy))  # expect ('unknown', ~something <= 0.30)
