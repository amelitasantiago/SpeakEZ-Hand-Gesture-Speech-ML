"""
Preprocessing module for hand detection and image processing
"""
import cv2
import numpy as np
from pathlib import Path

class HandDetector:
    """Detect and isolate hands using OpenCV"""
    
    def __init__(self, config):
        self.config = config
        self.target_size = tuple(config['preprocessing']['target_size'])
        self.skin_lower = np.array(config['preprocessing']['skin_hsv_lower'])
        self.skin_upper = np.array(config['preprocessing']['skin_hsv_upper'])
        self.kernel_size = config['preprocessing']['morph_kernel_size']
        self.padding = config['preprocessing']['padding']
        
        # Create morphological kernel
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.kernel_size, self.kernel_size)
        )
    
    def detect_hand(self, frame):
        """
        Detect hand in frame using skin color segmentation
        
        Args:
            frame: BGR image (H, W, 3)
        
        Returns:
            hand_roi: Extracted hand region or None if not detected
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create skin mask
        mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        
        # Morphological operations to clean up mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.dilate(mask, self.kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None, None
        
        # Get largest contour (assume it's the hand)
        hand_contour = max(contours, key=cv2.contourArea)
        
        # Filter out small contours (noise)
        if cv2.contourArea(hand_contour) < 1000:
            return None, None
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(hand_contour)
        
        # Add padding
        x1 = max(0, x - self.padding)
        y1 = max(0, y - self.padding)
        x2 = min(frame.shape[1], x + w + self.padding)
        y2 = min(frame.shape[0], y + h + self.padding)
        
        # Extract ROI
        hand_roi = frame[y1:y2, x1:x2]
        
        return hand_roi, (x1, y1, x2, y2)
    
    def preprocess(self, frame):
        """
        Full preprocessing pipeline: detect hand and prepare for model
        
        Args:
            frame: BGR image
        
        Returns:
            processed: Normalized image ready for model (192, 192, 3)
        """
        hand_roi, bbox = self.detect_hand(frame)
        
        if hand_roi is None:
            return None, None
        
        # Resize to target size
        resized = cv2.resize(hand_roi, self.target_size)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized, bbox
    
    def preprocess_batch(self, frames):
        """Preprocess batch of frames"""
        processed = []
        bboxes = []
        
        for frame in frames:
            proc, bbox = self.preprocess(frame)
            if proc is not None:
                processed.append(proc)
                bboxes.append(bbox)
        
        if processed:
            return np.array(processed), bboxes
        return None, None


def prepare_dataset(data_path, output_path, config):
    """
    Prepare dataset: load images, apply preprocessing, create splits
    
    Args:
        data_path: Path to raw ASL dataset
        output_path: Path to save processed data
        config: Configuration dictionary
    """
    from sklearn.model_selection import train_test_split
    import os
    from tqdm import tqdm
    
    print("Preparing dataset...")
    detector = HandDetector(config)
    
    # Get all class folders
    class_names = config['classes']
    X, y = [], []
    
    for class_idx, class_name in enumerate(tqdm(class_names, desc="Processing classes")):
        class_path = Path(data_path) / class_name
        if not class_path.exists():
            print(f"Warning: {class_name} folder not found")
            continue
        
        # Process each image in class folder
        for img_path in class_path.glob('*.jpg'):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # If dataset already has isolated hands, just resize
            # Otherwise, use hand detection
            processed = cv2.resize(img, tuple(config['preprocessing']['target_size']))
            processed = processed.astype(np.float32) / 255.0
            
            X.append(processed)
            y.append(class_idx)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")
    
    # Create train/val/test splits
    train_ratio = config['data']['train_split']
    val_ratio = config['data']['val_split']
    test_ratio = config['data']['test_split']
    
    # First split: train and temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        train_size=train_ratio, 
        random_state=42, 
        stratify=y
    )
    
    # Second split: val and test
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_size,
        random_state=42,
        stratify=y_temp
    )
    
    # Save splits
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / 'X_train.npy', X_train)
    np.save(output_path / 'y_train.npy', y_train)
    np.save(output_path / 'X_val.npy', X_val)
    np.save(output_path / 'y_val.npy', y_val)
    np.save(output_path / 'X_test.npy', X_test)
    np.save(output_path / 'y_test.npy', y_test)
    
    print(f"✓ Dataset prepared and saved to {output_path}")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == '__main__':
    """Run preprocessing as standalone script"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from utils import load_config
    
    print("="*70)
    print("SPEAKEZ PREPROCESSING")
    print("="*70)
    
    config = load_config()
    
    # Prepare dataset
    prepare_dataset(
        data_path='data/raw',
        output_path='data/splits',
        config=config
    )
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    print("\nNext step: python -m src.train")