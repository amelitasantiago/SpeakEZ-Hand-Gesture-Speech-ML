# src/hybrid_detector.py
"""
Smart Hybrid Detector: Combines letter and word recognition
- Continuously monitors for complete word gestures
- Falls back to letter-by-letter spelling
- Intelligent switching based on confidence
"""

import cv2
import numpy as np
from collections import deque
from typing import Tuple, Optional, Dict
import time

# Import both detection systems
from inference_ import predict_letter, LetterBuffer, LETTER_CLASSES
try:
    from word_detector import classify_word_video, WORDS
    WORD_DETECTION_AVAILABLE = True
except:
    WORD_DETECTION_AVAILABLE = False
    WORDS = []

class HybridDetector:
    """
    Smart hybrid detector that combines letter and word recognition.
    
    Strategy:
    1. Continuously run letter detection (fast, per-frame)
    2. Buffer recent frames for potential word detection
    3. Detect "gesture boundaries" (start/end of signing)
    4. When gesture completes, run word classifier
    5. If word confidence high → use word, else use buffered letters
    """
    
    def __init__(self, 
                 letter_conf_threshold=0.20,
                 word_conf_threshold=0.50,
                 word_buffer_frames=60,  # 2 seconds at 30fps
                 gesture_threshold=5,     # Frames of NOTHING to mark gesture end
                 letter_smooth_n=3):
        
        self.letter_threshold = letter_conf_threshold
        self.word_threshold = word_conf_threshold
        
        # Letter detection
        self.letter_buffer = LetterBuffer(N=letter_smooth_n)
        
        # Word detection
        self.word_detection_enabled = WORD_DETECTION_AVAILABLE
        self.frame_buffer = deque(maxlen=word_buffer_frames)
        self.gesture_active = False
        self.nothing_count = 0
        self.gesture_threshold = gesture_threshold
        
        # State tracking
        self.current_mode = "letter"  # "letter" or "word"
        self.last_word_time = 0
        self.word_cooldown = 1.0  # seconds
        
        # Statistics
        self.stats = {
            "letter_predictions": 0,
            "word_predictions": 0,
            "word_detections": 0,
            "frames_processed": 0
        }
    
    def process_frame(self, frame_bgr: np.ndarray) -> Dict:
        """
        Process a single frame and return detection results.
        
        Returns:
            dict with keys:
                - token: detected letter or word
                - confidence: confidence score
                - mode: "letter" or "word"
                - gesture_active: bool
                - buffer_updated: bool (if text buffer changed)
        """
        self.stats["frames_processed"] += 1
        
        # Always run letter detection (fast)
        letter_token, letter_conf = predict_letter(
            frame_bgr, 
            conf_threshold=self.letter_threshold
        )
        
        # Track gesture state
        self._update_gesture_state(letter_token)
        
        # Buffer frame for potential word detection
        if self.word_detection_enabled:
            self.frame_buffer.append(frame_bgr.copy())
        
        # Check if gesture just completed
        word_result = None
        if self.gesture_active and self.nothing_count >= self.gesture_threshold:
            if self.word_detection_enabled:
                word_result = self._try_word_detection()
        
        # Decide which result to use
        if word_result and word_result["confidence"] >= self.word_threshold:
            # Use word detection
            self.stats["word_predictions"] += 1
            self.stats["word_detections"] += 1
            self.letter_buffer.buf += word_result["word"] + " "
            
            return {
                "token": word_result["word"],
                "confidence": word_result["confidence"],
                "mode": "word",
                "gesture_active": self.gesture_active,
                "buffer_updated": True,
                "all_word_probs": word_result.get("all_probs", {})
            }
        else:
            # Use letter detection
            self.stats["letter_predictions"] += 1
            buffer_updated = self.letter_buffer.push(letter_token)
            
            return {
                "token": letter_token,
                "confidence": letter_conf,
                "mode": "letter",
                "gesture_active": self.gesture_active,
                "buffer_updated": buffer_updated,
                "all_word_probs": word_result.get("all_probs", {}) if word_result else {}
            }
    
    def _update_gesture_state(self, token: str):
        """Track whether a gesture is currently being performed"""
        if token in ["NOTHING", "unknown"]:
            self.nothing_count += 1
            if self.nothing_count >= self.gesture_threshold:
                if self.gesture_active:
                    # Gesture just ended
                    self.gesture_active = False
        else:
            # Active signing
            if self.nothing_count >= self.gesture_threshold:
                # New gesture starting
                self.gesture_active = True
            self.nothing_count = 0
    
    def _try_word_detection(self) -> Optional[Dict]:
        """
        Attempt word detection on buffered frames.
        Returns None if detection fails or confidence too low.
        """
        # Cooldown to prevent repeated detections
        if time.time() - self.last_word_time < self.word_cooldown:
            return None
        
        if len(self.frame_buffer) < 24:  # Need at least 24 frames
            return None
        
        try:
            # Save buffered frames as temporary video
            import tempfile
            import os
            
            # Get last 24-60 frames (1-2 seconds of gesture)
            frames = list(self.frame_buffer)[-60:]  # Max 2 seconds
            
            # Create temporary video file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
            os.close(temp_fd)
            
            # Write frames to video
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, 30, (w, h))
            
            for frame in frames:
                out.write(frame)
            out.release()
            
            # Run word detection
            from word_detector import classify_word_video
            label, conf, all_probs = classify_word_video(
                temp_path,
                thr=self.word_threshold * 0.8,  # Slightly lower for detection
                margin=0.05,
                agreement=False
            )
            
            # Clean up
            os.unlink(temp_path)
            
            self.last_word_time = time.time()
            
            if label != "unknown":
                return {
                    "word": label.upper(),  # Uppercase for consistency
                    "confidence": conf,
                    "all_probs": all_probs
                }
            
        except Exception as e:
            print(f"Word detection error: {e}")
        
        return None
    
    def get_buffer_text(self) -> str:
        """Get current text buffer"""
        return self.letter_buffer.buf
    
    def clear_buffer(self):
        """Clear text buffer"""
        self.letter_buffer.buf = ""
        self.letter_buffer.last = None
        self.letter_buffer.deq.clear()
    
    def get_stats(self) -> Dict:
        """Get detection statistics"""
        return self.stats.copy()


class SimpleHybridDetector:
    """
    Simplified hybrid detector for better performance.
    - No video buffering (too slow)
    - Uses hand pose/motion patterns to detect word gestures
    - Falls back to letters for everything else
    """
    
    def __init__(self,
                 letter_conf_threshold=0.20,
                 letter_smooth_n=3):
        
        self.letter_threshold = letter_conf_threshold
        self.letter_buffer = LetterBuffer(N=letter_smooth_n)
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "predictions": 0
        }
    
    def process_frame(self, frame_bgr: np.ndarray) -> Dict:
        """
        Simplified processing - letter detection only.
        Word detection via separate video upload UI.
        """
        self.stats["frames_processed"] += 1
        
        # Letter detection
        token, conf = predict_letter(
            frame_bgr,
            conf_threshold=self.letter_threshold
        )
        
        if token != "unknown":
            self.stats["predictions"] += 1
        
        buffer_updated = self.letter_buffer.push(token)
        
        return {
            "token": token,
            "confidence": conf,
            "mode": "letter",
            "buffer_updated": buffer_updated
        }
    
    def get_buffer_text(self) -> str:
        return self.letter_buffer.buf
    
    def clear_buffer(self):
        self.letter_buffer.buf = ""
        self.letter_buffer.last = None
        self.letter_buffer.deq.clear()
    
    def get_stats(self) -> Dict:
        return self.stats.copy()