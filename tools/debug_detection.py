"""
Debug script to find why detection isn't working
This will show you EXACTLY what's happening at each step
"""

import cv2
import numpy as np
import sys
from pathlib import Path
# Add parent directory's src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


print("="*70)
print("SpeakEZ Detection Debug Tool")
print("="*70)

# Test 1: Import modules
print("\n[1/6] Testing imports...")
try:
    from inference_ import predict_letter, LETTER_CLASSES, _letters_model, _preprocess_bgr
    print(f"✓ Letter model loaded: {_letters_model.input_shape} → {_letters_model.output_shape}")
    print(f"✓ Letter classes: {len(LETTER_CLASSES)} classes")
    print(f"  Sample: {LETTER_CLASSES[:3]} ... {LETTER_CLASSES[-3:]}")
except Exception as e:
    print(f"✗ Failed to import inference_: {e}")
    sys.exit(1)

try:
    from word_detector import LR, WORDS
    print(f"✓ Word model loaded: {LR.n_features_in_} features → {len(WORDS)} classes")
    print(f"✓ Word classes: {WORDS}")
except Exception as e:
    print(f"✗ Failed to import word_detector: {e}")
    print("  (Word detection will be skipped)")
    WORDS = None

# Test 2: Check camera
print("\n[2/6] Testing camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("✗ Cannot open camera 0, trying camera 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("✗ No camera found!")
        sys.exit(1)

ret, test_frame = cap.read()
if not ret:
    print("✗ Cannot read from camera!")
    sys.exit(1)

print(f"✓ Camera working: {test_frame.shape}")
cv2.imwrite('debug_camera_frame.jpg', test_frame)
print("  Saved test frame as 'debug_camera_frame.jpg'")

# Test 3: Test MediaPipe hand detection
print("\n[3/6] Testing MediaPipe hand detection...")
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )
    
    # Test on camera frame
    rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(rgb)
    
    if results.multi_hand_landmarks:
        print(f"✓ MediaPipe detected {len(results.multi_hand_landmarks)} hand(s)")
        print("  Note: Make sure your hand is visible to the camera!")
    else:
        print("⚠️  No hands detected in test frame (this is OK if no hand was visible)")
        print("  Tip: Position your hand in frame and try again")
    
except Exception as e:
    print(f"✗ MediaPipe error: {e}")
    sys.exit(1)

# Test 4: Test preprocessing
print("\n[4/6] Testing preprocessing...")
try:
    preprocessed = _preprocess_bgr(test_frame)
    print(f"✓ Preprocessing OK: {test_frame.shape} → {preprocessed.shape}")
    print(f"  Value range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
    
    # Save preprocessed image for inspection
    vis = (preprocessed * 255).astype(np.uint8)
    cv2.imwrite('debug_preprocessed.jpg', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print("  Saved preprocessed image as 'debug_preprocessed.jpg'")
    
except Exception as e:
    print(f"✗ Preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test model prediction with different thresholds
print("\n[5/6] Testing model prediction...")
try:
    # Get raw predictions
    x = preprocessed[None, ...]
    probs = _letters_model(x, training=False).numpy()[0]
    
    # Show top 5 predictions
    top5_idx = np.argsort(probs)[-5:][::-1]
    print(f"✓ Model inference successful")
    print(f"\n  Top 5 predictions:")
    for i, idx in enumerate(top5_idx):
        print(f"    {i+1}. {LETTER_CLASSES[idx]:10s} confidence={probs[idx]:.3f} ({probs[idx]*100:.1f}%)")
    
    # Test different thresholds
    print(f"\n  Testing different confidence thresholds:")
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        token, conf = predict_letter(test_frame, conf_threshold=threshold)
        status = "✓" if token != "unknown" else "✗"
        print(f"    {status} threshold={threshold:.2f} → prediction='{token}' (conf={conf:.3f})")
    
except Exception as e:
    print(f"✗ Model prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Live detection test
print("\n[6/6] Live detection test...")
print("\nStarting live camera feed with debugging...")
print("Controls:")
print("  Press 'q' to quit")
print("  Press 's' to save current frame for analysis")
print("  Press '+' to increase confidence threshold")
print("  Press '-' to decrease confidence threshold")
print("\n" + "="*70)

conf_threshold = 0.20
frame_count = 0
detection_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Get prediction
        token, conf = predict_letter(frame, conf_threshold=conf_threshold)
        
        if token != "unknown":
            detection_count += 1
        
        # Create display
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Background for text
        cv2.rectangle(display, (0, 0), (w, 150), (0, 0, 0), -1)
        
        # Show prediction with color coding
        if conf > 0.6:
            color = (0, 255, 0)  # Green - very confident
        elif conf > 0.4:
            color = (0, 255, 255)  # Yellow - medium
        elif conf > 0.2:
            color = (0, 165, 255)  # Orange - low
        else:
            color = (0, 0, 255)  # Red - very low
        
        cv2.putText(display, f"Prediction: {token}", (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(display, f"Confidence: {conf:.3f} ({conf*100:.1f}%)", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, f"Threshold: {conf_threshold:.2f} (use +/- to adjust)", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display, f"Detections: {detection_count}/{frame_count} ({detection_count/frame_count*100:.1f}%)", (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Show frame
        cv2.imshow('Debug - Letter Detection', display)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'debug_frame_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"\n✓ Saved frame as '{filename}'")
            print(f"  Prediction: {token} (conf={conf:.3f})")
        elif key == ord('+') or key == ord('='):
            conf_threshold = min(1.0, conf_threshold + 0.05)
            print(f"\nThreshold increased to {conf_threshold:.2f}")
        elif key == ord('-') or key == ord('_'):
            conf_threshold = max(0.0, conf_threshold - 0.05)
            print(f"\nThreshold decreased to {conf_threshold:.2f}")
        
        # Print prediction every 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {token} (conf={conf:.3f})")
    
except KeyboardInterrupt:
    print("\n\nInterrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("Debug Session Summary")
    print("="*70)
    print(f"Total frames: {frame_count}")
    print(f"Detections: {detection_count} ({detection_count/frame_count*100:.1f}%)")
    print(f"Final threshold: {conf_threshold:.2f}")
    
    if detection_count == 0:
        print("\n⚠️  WARNING: No detections at all!")
        print("Possible issues:")
        print("  1. Confidence threshold too high")
        print("  2. Hand not visible to camera")
        print("  3. Poor lighting conditions")
        print("  4. Model not trained correctly")
        print("\nRecommendations:")
        print("  - Try lowering threshold to 0.10 or 0.05")
        print("  - Ensure good lighting and plain background")
        print("  - Check if preprocessed images look correct")
    elif detection_count / frame_count < 0.3:
        print("\n⚠️  Low detection rate")
        print("  Consider lowering confidence threshold")
    else:
        print("\n✓ Detection rate looks good!")
    
    print("\nGenerated debug files:")
    print("  - debug_camera_frame.jpg (raw camera input)")
    print("  - debug_preprocessed.jpg (after preprocessing)")
    print("  - debug_frame_*.jpg (any saved frames)")
    print("\n" + "="*70)