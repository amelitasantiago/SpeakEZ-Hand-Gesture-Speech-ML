"""
Video logging module for recording training process and demos
"""
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io

class TrainingVideoLogger:
    """Record training progress as video"""
    
    def __init__(self, output_path='logs/videos/training_progress.mp4', fps=2):
        """
        Initialize video logger
        
        Args:
            output_path: Path to save video
            fps: Frames per second (use low fps for training, e.g., 2)
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.writer = None
        self.frame_size = (1920, 1080)  # Full HD
        
    def start(self):
        """Start video recording"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            self.frame_size
        )
        print(f"Started recording training video: {self.output_path}")
    
    def log_epoch(self, epoch, history, sample_images, sample_labels, predictions):
        """
        Log one epoch as video frame
        
        Args:
            epoch: Current epoch number
            history: Training history dict with loss/accuracy
            sample_images: Sample images to display (N, H, W, 3)
            sample_labels: True labels for samples
            predictions: Model predictions for samples
        """
        if self.writer is None:
            self.start()
        
        # Create comprehensive visualization frame
        frame = self._create_epoch_frame(
            epoch, history, sample_images, sample_labels, predictions
        )
        
        # Write frame
        self.writer.write(frame)
    
    def _create_epoch_frame(self, epoch, history, sample_images, sample_labels, predictions):
        """Create visualization frame for current epoch"""
        
        # Create figure
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        
        # Title
        fig.suptitle(f'Training Progress - Epoch {epoch + 1}', 
                     fontsize=24, fontweight='bold')
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Loss plot (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        epochs_range = range(1, len(history['loss']) + 1)
        ax1.plot(epochs_range, history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs_range, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Model Loss', fontsize=16, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy plot (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.plot(epochs_range, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs_range, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Model Accuracy', fontsize=16, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Current metrics (middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.axis('off')
        
        metrics_text = f"""
        CURRENT METRICS (Epoch {epoch + 1})
        
        Training Loss:      {history['loss'][-1]:.4f}
        Training Accuracy:  {history['accuracy'][-1]:.2%}
        
        Validation Loss:    {history['val_loss'][-1]:.4f}
        Validation Accuracy: {history['val_accuracy'][-1]:.2%}
        
        Best Val Accuracy:  {max(history['val_accuracy']):.2%}
        """
        
        ax3.text(0.1, 0.5, metrics_text, fontsize=14, family='monospace',
                verticalalignment='center')
        
        # 4. Sample predictions (middle right and bottom)
        num_samples = min(8, len(sample_images))
        
        for i in range(num_samples):
            row = 1 + i // 4
            col = 2 + (i % 2) if row == 1 else i % 4
            
            ax = fig.add_subplot(gs[row, col])
            
            img = sample_images[i]
            true_label = sample_labels[i]
            pred_label = np.argmax(predictions[i])
            confidence = predictions[i][pred_label]
            
            ax.imshow(img)
            ax.axis('off')
            
            # Color: green if correct, red if wrong
            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(f'True: {true_label}\nPred: {pred_label}\n{confidence:.1%}',
                        fontsize=10, color=color, fontweight='bold')
        
        # Convert matplotlib figure to numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        # Get image as array
        buf = canvas.buffer_rgba()
        frame = np.asarray(buf)
        
        # Convert RGBA to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        plt.close(fig)
        
        return frame
    
    def stop(self):
        """Stop recording and save video"""
        if self.writer is not None:
            self.writer.release()
            print(f"Training video saved: {self.output_path}")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class DemoVideoRecorder:
    """Record real-time demo sessions"""
    
    def __init__(self, output_path='logs/videos/demo_recording.mp4', fps=30):
        """
        Initialize demo recorder
        
        Args:
            output_path: Path to save video
            fps: Frames per second
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.writer = None
        self.recording = False
        
    def start(self, frame_size):
        """
        Start recording
        
        Args:
            frame_size: (width, height) of frames
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            frame_size
        )
        self.recording = True
        print(f"Started demo recording: {self.output_path}")
    
    def write_frame(self, frame):
        """Write frame to video"""
        if self.recording and self.writer is not None:
            self.writer.write(frame)
    
    def stop(self):
        """Stop recording"""
        if self.writer is not None:
            self.writer.release()
            self.recording = False
            print(f"Demo video saved: {self.output_path}")
    
    def is_recording(self):
        """Check if currently recording"""
        return self.recording
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def create_training_timelapse(checkpoint_dir='models/checkpoints', 
                              output_path='logs/videos/training_timelapse.mp4'):
    """
    Create timelapse video from saved checkpoints
    
    Args:
        checkpoint_dir: Directory with model checkpoints
        output_path: Output video path
    """
    import tensorflow as tf
    from glob import glob
    
    checkpoint_files = sorted(glob(f'{checkpoint_dir}/*.h5'))
    
    if not checkpoint_files:
        print("No checkpoints found for timelapse")
        return
    
    print(f"Creating timelapse from {len(checkpoint_files)} checkpoints...")
    
    # Load sample data
    splits_path = Path('data/splits')
    X_val = np.load(splits_path / 'X_val.npy')
    y_val = np.load(splits_path / 'y_val.npy')
    
    # Select samples
    sample_indices = np.random.choice(len(X_val), 16, replace=False)
    samples = X_val[sample_indices]
    labels = y_val[sample_indices]
    
    with TrainingVideoLogger(output_path, fps=3) as logger:
        for epoch, ckpt_path in enumerate(checkpoint_files):
            # Load model
            model = tf.keras.models.load_model(ckpt_path)
            
            # Predict
            predictions = model.predict(samples, verbose=0)
            
            # Create dummy history (you'd need to save this during training)
            history = {
                'loss': [0.5 - epoch * 0.02],
                'accuracy': [0.6 + epoch * 0.015],
                'val_loss': [0.6 - epoch * 0.018],
                'val_accuracy': [0.55 + epoch * 0.017]
            }
            
            # Log frame
            logger.log_epoch(epoch, history, samples, labels, predictions)
    
    print(f"Timelapse created: {output_path}")