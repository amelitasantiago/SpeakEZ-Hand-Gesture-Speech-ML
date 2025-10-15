"""
Simple model sanity check - verify model is making predictions
Run this FIRST before anything else!

Usage: python model_sanity_check.py
"""

import numpy as np
import sys

print("="*70)
print("MODEL SANITY CHECK")
print("="*70)

# Test 1: Can we import?
print("\n[1/4] Testing imports...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow: {tf.__version__}")
except Exception as e:
    print(f"✗ TensorFlow import failed: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"✗ OpenCV import failed: {e}")
    sys.exit(1)

# Test 2: Can we load the model?
print("\n[2/4] Loading model...")
try:
    from pathlib import Path
    model_path = Path("models/final/asl_baseline_cnn_128_final.h5")
    
    if not model_path.exists():
        print(f"✗ Model file not found: {model_path}")
        print(f"  Current directory: {Path.cwd()}")
        print(f"  Make sure you're in the project root!")
        sys.exit(1)
    
    model = tf.keras.models.load_model(str(model_path), compile=False)
    print(f"✓ Model loaded from: {model_path}")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    
    # Verify output shape
    expected_classes = 29  # A-Z + DEL + SPACE + NOTHING
    actual_classes = model.output_shape[-1]
    
    if actual_classes == expected_classes:
        print(f"✓ Output classes: {actual_classes} (correct!)")
    else:
        print(f"⚠️  Output classes: {actual_classes} (expected {expected_classes})")
        print(f"  Your model might have different number of classes!")
    
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Can we make a prediction?
print("\n[3/4] Testing prediction with random image...")
try:
    # Create random RGB image (simulating camera input)
    random_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    
    # Preprocess like inference_.py does
    img_float = random_img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img_batch = img_float[None, ...]  # Add batch dimension (1, 128, 128, 3)
    
    print(f"  Input shape: {img_batch.shape}")
    print(f"  Input range: [{img_batch.min():.3f}, {img_batch.max():.3f}]")
    
    # Predict
    probs = model(img_batch, training=False).numpy()[0]
    
    print(f"  Output shape: {probs.shape}")
    print(f"  Output range: [{probs.min():.6f}, {probs.max():.6f}]")
    print(f"  Output sum: {probs.sum():.6f} (should be ~1.0 for softmax)")
    
    # Show top 5 predictions
    top5_idx = np.argsort(probs)[-5:][::-1]
    print(f"\n  Top 5 predictions (random image):")
    
    # Load class names
    import json
    classes_path = Path("models/final/classes_letters.json")
    if classes_path.exists():
        with open(classes_path) as f:
            classes = json.load(f)
    else:
        classes = [chr(i) for i in range(ord('A'), ord('Z')+1)] + ["DEL", "SPACE", "NOTHING"]
    
    for i, idx in enumerate(top5_idx, 1):
        print(f"    {i}. {classes[idx]:10s} = {probs[idx]:.4f} ({probs[idx]*100:.1f}%)")
    
    max_conf = probs[top5_idx[0]]
    if max_conf > 0.5:
        print(f"\n  ⚠️  Confidence {max_conf:.3f} is HIGH for random image!")
        print(f"  This might indicate overfitting.")
    elif max_conf < 0.01:
        print(f"\n  ⚠️  Confidence {max_conf:.3f} is VERY LOW!")
        print(f"  Model might not be trained properly.")
    else:
        print(f"\n  ✓ Confidence {max_conf:.3f} looks reasonable for random input")
    
except Exception as e:
    print(f"✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Can we make predictions with different inputs?
print("\n[4/4] Testing with multiple inputs...")
try:
    results = []
    
    for test_name, test_img in [
        ("all zeros", np.zeros((128, 128, 3), dtype=np.uint8)),
        ("all ones", np.ones((128, 128, 3), dtype=np.uint8) * 255),
        ("random 1", np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)),
        ("random 2", np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)),
    ]:
        img_batch = (test_img.astype(np.float32) / 255.0)[None, ...]
        probs = model(img_batch, training=False).numpy()[0]
        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        results.append((test_name, classes[top_idx], top_conf))
    
    print("\n  Test results:")
    for test_name, pred, conf in results:
        print(f"    {test_name:15s} → {pred:10s} (conf={conf:.3f})")
    
    # Check if predictions are too consistent (might indicate issue)
    predictions = [r[1] for r in results]
    unique_preds = len(set(predictions))
    
    if unique_preds == 1:
        print(f"\n  ⚠️  All predictions are the same: '{predictions[0]}'")
        print(f"  Model might always predict the same class!")
    else:
        print(f"\n  ✓ Got {unique_preds} different predictions (diversity is good)")
    
except Exception as e:
    print(f"✗ Multiple input test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nModel appears to be:")
print("  ✓ Loadable")
print("  ✓ Executable")
print("  ✓ Producing outputs")

print("\nNext steps:")
print("  1. Run: python debug_detection.py")
print("     This will test with actual camera input")
print("")
print("  2. If debug_detection shows low confidences (<0.20):")
print("     - Lower threshold in inference_.py")
print("     - Improve lighting/background")
print("     - Check if model needs retraining")
print("")
print("  3. Try simple camera test:")
print("     python test_camera.py  (or app_simple.py)")

print("\n" + "="*70)
print("✓ SANITY CHECK PASSED")
print("="*70)