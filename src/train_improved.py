"""
Training pipeline for SpeakEZ ASL gesture recognition
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from pathlib import Path
import json
from datetime import datetime
import time

from src.model import build_model, compile_model, get_callbacks, print_model_summary
from src.utils import load_config, create_directories, calculate_class_weights, Logger
from src.video_logger import TrainingVideoLogger
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

class Trainer:
    """Training pipeline manager"""
    
    def __init__(self, config_path='config/config.yaml'):
        self.config = load_config(config_path)
        self.logger = Logger()
        create_directories(self.config)
        self.training_start_time = None
        self.training_end_time = None
        
    def load_data(self):
        """Load preprocessed data splits"""
        splits_path = Path(self.config['data']['splits_path'])
        
        print(f"\n{'='*70}")
        print("LOADING PREPROCESSED DATA")
        print(f"{'='*70}")
        
        load_start = time.time()
        
        self.logger.log("Loading data splits from disk...")
        self.X_train = np.load(splits_path / 'X_train.npy')
        print(f"  ✓ Training data loaded: {self.X_train.shape}")
        
        self.y_train = np.load(splits_path / 'y_train.npy')
        print(f"  ✓ Training labels loaded: {self.y_train.shape}")
        
        self.X_val = np.load(splits_path / 'X_val.npy')
        print(f"  ✓ Validation data loaded: {self.X_val.shape}")
        
        self.y_val = np.load(splits_path / 'y_val.npy')
        print(f"  ✓ Validation labels loaded: {self.y_val.shape}")
        
        self.X_test = np.load(splits_path / 'X_test.npy')
        print(f"  ✓ Test data loaded: {self.X_test.shape}")
        
        self.y_test = np.load(splits_path / 'y_test.npy')
        print(f"  ✓ Test labels loaded: {self.y_test.shape}")
        
        load_time = time.time() - load_start
        print(f"\nData loaded in {load_time:.1f} seconds")
        
        # Convert labels to categorical
        num_classes = self.config['model']['num_classes']
        self.y_train_cat = to_categorical(self.y_train, num_classes)
        self.y_val_cat = to_categorical(self.y_val, num_classes)
        self.y_test_cat = to_categorical(self.y_test, num_classes)
        
        # Print class distribution
        unique, counts = np.unique(self.y_train, return_counts=True)
        print(f"\nTraining set class distribution:")
        class_names = self.config['classes']
        for cls_idx, count in zip(unique[:5], counts[:5]):  # Show first 5
            print(f"  {class_names[cls_idx]:8s}: {count:4d} samples")
        print(f"  ... (showing 5 of {len(unique)} classes)")
        
        print(f"{'='*70}\n")
        
        return self
    
    def create_data_generators(self):
        """Create augmented data generators"""
        aug_config = self.config['augmentation']
        
        # Training generator with augmentation
        self.train_datagen = ImageDataGenerator(
            rotation_range=aug_config['rotation_range'],
            width_shift_range=aug_config['width_shift_range'],
            height_shift_range=aug_config['height_shift_range'],
            brightness_range=aug_config['brightness_range'],
            zoom_range=aug_config['zoom_range'],
            horizontal_flip=False,  # Don't flip ASL signs!
            fill_mode='nearest'
        )
        
        # Validation generator (no augmentation)
        self.val_datagen = ImageDataGenerator()
        
        batch_size = self.config['training']['batch_size']
        
        self.train_generator = self.train_datagen.flow(
            self.X_train, self.y_train_cat,
            batch_size=batch_size,
            shuffle=True
        )
        
        self.val_generator = self.val_datagen.flow(
            self.X_val, self.y_val_cat,
            batch_size=batch_size,
            shuffle=False
        )
        
        self.logger.log("Data generators created with augmentation")
        return self
    
    def build_and_compile_model(self):
        """Build and compile the model"""
        self.logger.log("Building model...")
        self.model = build_model(self.config)
        self.model = compile_model(self.model, self.config)
        print_model_summary(self.model)
        return self
    
    def train(self):
        """Execute training with video logging"""
        self.training_start_time = datetime.now()
        
        print(f"\n{'='*70}")
        print("TRAINING STARTED")
        print(f"Start Time: {self.training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        self.logger.log("Initializing training...")
        
        # Calculate class weights for imbalanced data
        class_weights = calculate_class_weights(
            self.y_train, 
            self.config['model']['num_classes']
        )
        
        # Get callbacks
        callbacks = get_callbacks(self.config)
        
        # Initialize video logger
        #video_logger = TrainingVideoLogger(
            #output_path='logs/videos/training_progress.mp4',
            #fps=2  # 2 frames per second for training video
        #)
        #video_logger.start()
        video_logger = None  # Add this line
        
        # Select sample images for video visualization
        sample_indices = np.random.choice(len(self.X_val), 8, replace=False)
        sample_images = self.X_val[sample_indices]
        sample_labels = self.y_val[sample_indices]
        
        # Custom callback for video logging and progress tracking
        class VideoCallback(tf.keras.callbacks.Callback):
            def __init__(self, video_logger, sample_images, sample_labels, trainer):
                super().__init__()
                #self.video_logger = video_logger
                self.sample_images = sample_images
                self.sample_labels = sample_labels
                self.trainer = trainer
                self.history_dict = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
                self.epoch_start_time = None
            
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()
                print(f"\n{'─'*70}")
                print(f"Epoch {epoch + 1}/{self.trainer.config['training']['epochs']}")
                print(f"{'─'*70}")
            
            def on_epoch_end(self, epoch, logs=None):
                # Update history
                self.history_dict['loss'].append(logs['loss'])
                self.history_dict['accuracy'].append(logs['accuracy'])
                self.history_dict['val_loss'].append(logs['val_loss'])
                self.history_dict['val_accuracy'].append(logs['val_accuracy'])
                
                # Calculate epoch duration
                epoch_duration = time.time() - self.epoch_start_time
                
                # Print epoch summary
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"  Train Loss: {logs['loss']:.4f} | Train Acc: {logs['accuracy']:.4f}")
                print(f"  Val Loss:   {logs['val_loss']:.4f} | Val Acc:   {logs['val_accuracy']:.4f}")
                print(f"  Duration: {epoch_duration:.1f}s")
                
                # Get predictions for sample images
                predictions = self.model.predict(self.sample_images, verbose=0)
                
                # Log to video
                if self.video_logger:
                    self.video_logger.log_epoch(
                        epoch,
                        self.history_dict,
                        self.sample_images,
                        self.sample_labels,
                        predictions
                    )
        
        #video_callback = VideoCallback(video_logger, sample_images, sample_labels, self)
        video_callback = VideoCallback(sample_images, sample_labels, self)  # Removed video_logger
        callbacks.append(video_callback)
        
        # Train model
        epochs = self.config['training']['epochs']
        
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {self.config['training']['batch_size']}")
        print(f"  Learning rate: {self.config['training']['learning_rate']}")
        print(f"  Optimizer: AdamW")
        print(f"  Using class weights: Yes")
        print(f"  Video logging: Enabled\n")
        
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Stop video logging
        #video_logger.stop()
        
        # Calculate training duration
        self.training_end_time = datetime.now()
        training_duration = self.training_end_time - self.training_start_time
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETED")
        print(f"End Time: {self.training_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Duration: {training_duration.total_seconds()/60:.1f} minutes")
        print(f"  Best Val Accuracy: {max(self.history.history['val_accuracy']):.4f}")
        print(f"  Final Train Accuracy: {self.history.history['accuracy'][-1]:.4f}")
        print(f"{'='*70}\n")
        
        self.logger.log("Training completed!")
        return self
    
    def evaluate(self):
        """Evaluate model on test set"""
        self.logger.log("Evaluating on test set...")
        
        test_loss, test_acc, test_top3 = self.model.evaluate(
            self.X_test, self.y_test_cat,
            verbose=1
        )
        
        self.logger.log(f"Test Results:")
        self.logger.log(f"  Loss: {test_loss:.4f}")
        self.logger.log(f"  Accuracy: {test_acc:.4f}")
        self.logger.log(f"  Top-3 Accuracy: {test_top3:.4f}")
        
        # Per-class evaluation
        predictions = self.model.predict(self.X_test)
        pred_classes = np.argmax(predictions, axis=1)
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        class_names = self.config['classes']
        report = classification_report(
            self.y_test, pred_classes,
            target_names=class_names,
            output_dict=True
        )
        
        self.logger.log("\nPer-class F1 Scores:")
        for cls in class_names:
            if cls in report:
                self.logger.log(f"  {cls}: {report[cls]['f1-score']:.3f}")
        
        # Save results
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_top3_accuracy': float(test_top3),
            'classification_report': report,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('models/final/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return self
    
    def save_model(self):
        """Save final model"""
        self.logger.log("Saving final model...")
        
        # Save full model
        self.model.save('models/final/speakez_model.h5')
        
        # Save model weights only
        self.model.save_weights('models/final/speakez_weights.h5')
        
        # Save training history
        history_dict = {
            'loss': [float(x) for x in self.history.history['loss']],
            'accuracy': [float(x) for x in self.history.history['accuracy']],
            'val_loss': [float(x) for x in self.history.history['val_loss']],
            'val_accuracy': [float(x) for x in self.history.history['val_accuracy']]
        }
        
        with open('models/final/training_history.json', 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        self.logger.log("Model saved successfully!")
        return self
    
    def visualize_training(self):
        """Create training visualizations"""
        self.logger.log("Creating training visualizations...")
        
        # Create output directory
        Path('logs/visualizations').mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Training & Validation Loss
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Model Loss Over Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Training & Validation Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('logs/visualizations/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Confusion Matrix
        predictions = self.model.predict(self.X_test)
        pred_classes = np.argmax(predictions, axis=1)
        
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(self.y_test, pred_classes)
        
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.config['classes'],
                    yticklabels=self.config['classes'],
                    cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('logs/visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 4: Sample predictions
        self._visualize_sample_predictions()
        
        self.logger.log("Visualizations saved to logs/visualizations/")
        return self
    
    def _visualize_sample_predictions(self):
        """Visualize sample predictions"""
        # Select random samples from test set
        num_samples = 25
        indices = np.random.choice(len(self.X_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(5, 5, figsize=(15, 15))
        fig.suptitle('Sample Test Predictions', fontsize=16, fontweight='bold')
        
        predictions = self.model.predict(self.X_test[indices])
        
        for idx, ax in enumerate(axes.flat):
            img = self.X_test[indices[idx]]
            true_label = self.config['classes'][self.y_test[indices[idx]]]
            pred_label = self.config['classes'][np.argmax(predictions[idx])]
            confidence = predictions[idx][np.argmax(predictions[idx])]
            
            ax.imshow(img)
            ax.axis('off')
            
            # Color code: green if correct, red if wrong
            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}',
                        fontsize=8, color=color)
        
        plt.tight_layout()
        plt.savefig('logs/visualizations/sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def log_sample_data(self):
        """Log sample images from dataset for documentation"""
        self.logger.log("Logging sample data...")
        
        Path('logs/samples').mkdir(parents=True, exist_ok=True)
        
        # Save one sample per class
        for class_idx, class_name in enumerate(self.config['classes']):
            # Find first occurrence of this class
            mask = self.y_train == class_idx
            if not np.any(mask):
                continue
            
            sample_idx = np.where(mask)[0][0]
            sample_img = self.X_train[sample_idx]
            
            # Denormalize for saving
            img_to_save = (sample_img * 255).astype(np.uint8)
            img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(f'logs/samples/{class_name}.jpg', img_to_save)
        
        self.logger.log(f"Sample images saved to logs/samples/")
        return self

def main():
    """Main training execution"""
    print("\n" + "="*70)
    print("SPEAKEZ TRAINING PIPELINE")
    print("="*70 + "\n")
    
    trainer = Trainer()
    
    # Execute training pipeline
    (trainer
     .load_data()
     .log_sample_data()
     .create_data_generators()
     .build_and_compile_model()
     .train()
     .evaluate()
     .visualize_training()
     .save_model())
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()