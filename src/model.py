"""
CNN model architecture for ASL gesture recognition
"""
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, Input, BatchNormalization
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)

def build_model(config):
    """
    Build 3-layer CNN architecture as per slide 15
    
    Architecture:
    - Conv2D-32 → Conv2D-32 → MaxPool
    - Conv2D-64 → Conv2D-64 → MaxPool
    - Conv2D-128 → Conv2D-128 → GlobalAvgPool
    - Dense-512 → Dropout → Dense-29
    
    Args:
        config: Configuration dictionary
    
    Returns:
        model: Compiled Keras model
    """
    input_shape = tuple(config['model']['input_shape'])
    num_classes = config['model']['num_classes']
    dropout_rate = config['model']['dropout_rate']
    
    model = Sequential([
        # Input layer
        Input(shape=input_shape),
        
        # Block 1: 32 filters
        Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1'),
        Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
        MaxPooling2D((2, 2), name='pool1'),
        
        # Block 2: 64 filters
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
        MaxPooling2D((2, 2), name='pool2'),
        
        # Block 3: 128 filters
        Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
        Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'),
        GlobalAveragePooling2D(name='global_pool'),
        
        # Classifier
        Dense(512, activation='relu', name='fc1'),
        Dropout(dropout_rate, name='dropout'),
        Dense(num_classes, activation='softmax', name='output')
    ], name='SpeakEZ_CNN')
    
    return model

def compile_model(model, config):
    """
    Compile model with AdamW optimizer and loss function
    
    Args:
        model: Keras model
        config: Configuration dictionary
    
    Returns:
        model: Compiled model
    """
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    label_smoothing = config['training']['label_smoothing']
    
    # AdamW optimizer
    optimizer = AdamW(learning_rate=lr, weight_decay=weight_decay)
    
    # Loss with label smoothing
    loss = CategoricalCrossentropy(label_smoothing=label_smoothing)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )
    
    return model

def get_callbacks(config):
    """
    Create training callbacks
    
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath='models/checkpoints/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config['training']['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir='logs/tensorboard',
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    return callbacks

def print_model_summary(model):
    """Print model architecture summary"""
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE")
    print("="*70)
    model.summary()
    print("="*70 + "\n")
    
    # Calculate total parameters
    total_params = model.count_params()
    print(f"Total parameters: {total_params:,}")
    print(f"Model size estimate: {total_params * 4 / (1024**2):.2f} MB (float32)\n")
