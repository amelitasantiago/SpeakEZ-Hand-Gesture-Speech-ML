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

from src.model import build_model, compile_model, get_callbacks, print_model_summary
from src.utils import load_config, create_directories, calculate_class_weights, Logger

class Trainer:
    """Training pipeline manager"""
    
    def __init__(self, config_path='config/config.yaml'):
        self.config = load_config(config_path)
        self.logger = Logger()
        create_directories(self.config)
        
    def load_data(self):
        """Load preprocessed data splits"""
        splits_path = Path(self.config['data']['splits_path'])
        
        self.logger.log("Loading data splits...")
        self.X_train = np.load(splits_path / 'X_train.npy')
        self.y_train = np.load(splits_path / 'y_train.npy')
        self.X_val = np.load(splits_path / 'X_val.npy')
        self.y_val = np.load(splits_path / 'y_val.npy')
        self.X_test = np.load(splits_path / 'X_test.npy')
        self.y_test = np.load(splits_path / 'y_test.npy')
        
        self.logger.log(f"Train: {self.X_train.shape}, Val: {self.X_val.shape}, Test: {self.X_test.shape}")
        
        # Convert labels to categorical
        num_classes = self.config['model']['num_classes']
        self.y_train_cat = to_categorical(self.y_train, num_classes)
        self.y_val_cat = to_categorical(self.y_val, num_classes)
        self.y_test_cat = to_categorical(self.y_test, num_classes)
        
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
        """Execute training"""
        self.logger.log("Starting training...")
        
        # Calculate class weights for imbalanced data
        class_weights = calculate_class_weights(
            self.y_train, 
            self.config['model']['num_classes']
        )
        
        # Get callbacks
        callbacks = get_callbacks(self.config)
        
        # Train model
        epochs = self.config['training']['epochs']
        
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
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

def main():
    """Main training execution"""
    print("\n" + "="*70)
    print("SPEAKEZ TRAINING PIPELINE")
    print("="*70 + "\n")
    
    trainer = Trainer()
    
    # Execute training pipeline
    (trainer
     .load_data()
     .create_data_generators()
     .build_and_compile_model()
     .train()
     .evaluate()
     .save_model())
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()