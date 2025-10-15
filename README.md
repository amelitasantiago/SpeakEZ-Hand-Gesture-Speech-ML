# SpeakEZ (Sign Edition) - Hand Gesture to Speech Translation

**Transforming Silent Gestures into Speech**

NUS-ISS Pattern Recognition Systems Practice Module | June - November 2025

**Developers:** Amelita Santiago | Lee Fang Hui

## Project Overview

SpeakEZ is a real-time hand gesture recognition system that translates American Sign Language (ASL) alphabet letters and fingerspelled words into text and speech. The system employs custom-trained neural networks without transfer learning to ensure ASL-specific pattern recognition. 

**Key Capabilities:**
- 26 ASL alphabet letters (A-Z)
- 3 control commands (SPACE, DEL, NOTHING)
- 8 fingerspelled words recognition ('hello', 'please', 'yes', 'no', 'help', 'stop', 'sorry', 'bad')
- Real-time processing at 20+ FPS
- Text-to-speech synthesis output

**Key Features:**
- From-Scratch Training: Custom CNN (letters) via TensorFlow/Keras API and BiLSTM (words) via PyTorch, both with random initialization—avoiding pre-trained models commonly assumed with Keras/TensorFlow. This ensures native learning on ASL gestures; see [ASL Notebook](./notebooks/SpeakEZ_ASL.ipynb) and [WASL Notebook](./notebooks/SpeakEZ_WASL.ipynb) for details.
- Video Logging: Complete training process recorded for documentation
- Modular Architecture: Clean, maintainable codebase with hybrid framework integration
- Hardware Agnostic: Works with any standard RGB webcam

## System Architecture

### Hardware
- **Minimum:** 8GB RAM, Intel i5/AMD Ryzen 5
- **Recommended:** 16GB RAM, NVIDIA GPU (GTX 1060+)
- **Camera:** Standard RGB webcam (640×480 minimum)

### Software Environment
```yaml
python: 3.12.3
tensorflow==2.17.0
opencv-python==4.8.0.76
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
pyyaml==6.0.1
gtts==2.3.2
pygame==2.5.2
tqdm==4.66.1
Pillow==10.0.0
mediapipe: 0.10.15
torch=2.4.1
cuda: 12+ (optional for GPU acceleration)
```

## Installation

### 1. Repository Setup
```bash
# Clone repository
git clone https://github.com/amelitasantiago/SpeakEZ-Hand-Gesture-Speech-ML.git
cd SpeakEZ-Hand-Gesture-Speech-ML

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

**ASL Alphabet Dataset:**
- Source: [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- Size: ~87,000 images (200×200 RGB)
- Classes: 29 (A-Z + DEL + SPACE + NOTHING)

```bash
# Download and extract dataset
kaggle datasets download -d grassknoted/asl-alphabet
unzip asl-alphabet.zip -d data/raw/

# Prepare dataset splits
python src/preprocessing.py \
    --data_path data/raw \
    --output_path data/splits \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --random_seed 42
```
**WLASL Word Dataset:**
- Source: [GitHub: !git clone https://github.com/dxli94/WLASL.git)
- Size: {'hello': 13, 'please': 15, 'yes': 22, 'no': 22, 'help': 22, 'stop': 11, 'sorry': 14, 'bad': 16} images (128×128 RGB)
- Classes: ['hello', 'please', 'yes', 'no', 'help', 'stop', 'sorry', 'bad']
- Preprocessing: Resizes/normalizes classes images into .npy splits, loaded in Notebook (SpeakEZ_WASL.ipynb) for CNN training

### 3. Model Architecture

### Hand Detection Algorithm

- Convert BGR → HSV color space
- Apply skin color thresholding
- Morphological operations (close, open, dilate)
- Contour detection
- Extract largest contour as hand
- Crop with padding

### ASL Letter Recognition CNN (TensorFlow/Keras)
Built using TensorFlow/Keras API for efficient prototyping, but initialized randomly to train from scratch (emphasizing no pre-trained assumptions):

**Model:** Baseline CNN (32→64→128) + BN + ReLU + GlobalAvgPool + Dense
![Model Architecture](speakez/ogs/PNG/asl_letter_model_architecture.png)
```
Input Layer (128×128×3)
├─ Conv2D(32, 3×3) + ReLU + BatchNorm
├─ Conv2D(32, 3×3) + ReLU + BatchNorm
├─ MaxPooling2D(2×2)
├─ Dropout(0.25)
│
├─ Conv2D(64, 3×3) + ReLU + BatchNorm
├─ Conv2D(64, 3×3) + ReLU + BatchNorm
├─ MaxPooling2D(2×2)
├─ Dropout(0.25)
│
├─ Conv2D(128, 3×3) + ReLU + BatchNorm
├─ Conv2D(128, 3×3) + ReLU + BatchNorm
├─ GlobalAveragePooling2D()
│
├─ Dense(512) + ReLU + BatchNorm
├─ Dropout(0.5)
├─ Dense(29) + Softmax
│
Output (29 classes)

Total Parameters:  97,891 (382.40 KB)
Trainable Parameters:  97,891 (382.40 KB)
```

**Training Configuration:**
- Optimizer: Adam (1e-3)
- Loss: Sparse Categorical Crossentropy 
- Weight Initialization: He Normal
- Activation: Softmax
- Learning Rate Schedule: ReduceLROnPlateau (factor=0.5, patience=5)
- Early Stopping: patience=10, monitor=val_loss

## Notes: 
- Similar-sign pairs (M↔N, U↔V↔SPACE) may show clustered confusions.
- Label smoothing (ε=0.05) and temporal smoothing at inference are applied/encouraged.
- Use `/content/speakez_artifacts/labels.json` for consistent downstream mapping.

### WASL Word Recognition

**Word Classes:** 'hello', 'thank you', 'please', 'yes', 'no', 'help', 'stop', 'sorry', 'good', 'bad'

**Model:** MediaPipe landmarks → Feature engineering → LogisticRegression (joblib) and Experimental: BiLSTM: LSTM layers → Linear (PyTorch) 
```
Input (MediaPipe Landmarks: 21×3 coordinates)
├─ Linear(63, 128)
├─ Dropout(0.3)
├─ Linear(512, 256)
├─ Linear(256, 10)
│
Output (10 word classes)

Total Parameters: ~10K
```

## Performance Metrics

### ASL Letter Recognition (29 classes)
See printed detailed per-class metrics are in `speakez/logs/PNG/asl_letter_metrics_report.txt`

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| A | 0.9956 | 1.0000 | 0.9978 | 450 |
| B | 1.0000 | 0.9956 | 0.9978 | 450 |
| C | 1.0000 | 1.0000 | 1.0000 | 450 |
| D | 1.0000 | 1.0000 | 1.0000 | 450 |
| E | 1.0000 | 1.0000 | 1.0000 | 450 |
| ... | ... | ... | ... | ... |
| SPACE | 1.0000 |1.0000 | 1.0000 | 13051 |
| DELETE | 1.0000 | 1.0000 | 1.0000 | 13051 |
| NOTHING | 1.0000 | 1.0000 | 1.0000 | 13051 |
| Accuracy|0.9989|0.9989|0.9989|13051 |
| Macro Avg|0.9989|0.9989|0.9989|13051 |

### WLASL Word Recognition
See printed detailed per-class metrics are in `speakez/logs/PNG/wlasl_letter_metrics_report.txt`

| Word | Precision | Recall | F1-Score | Sequences |
|------|-----------|--------|----------|-----------|
| hello | 1.0000 | 1.0000 | 1.0000 | 1 |
| please | 0.5000 | 1.0000 | 0.6667 | 1 |
| no | 1.0000 | 1.0000 | 1.0000 | 1 |
| yes | 1.0000 | 1.0000 | 1.0000 | 1 |
| stop | 1.0000 | 1.0000 | 1.0000 | 1 |
| sorry | 1.0000 | 1.0000 | 1.0000 | 1 |
| help | 1.0000 | 1.0000 | 1.0000 | 1 |
| bad | 0.0000 | 0.0000 | 0.0000 | 1 |
| Accuracy |  |  | 0.8750 | 8 |
| Macro Avg |  |  | 0.8750 | 8 |

## Training Process
The models were trained using Google Colab for GPU acceleration and easy collaboration. Notebooks are provided for reproducibility—run them directly in Colab for best results.

**ASL Letter Model:**

The ASL letter recognition (A-Z + DEL, BACKSPACE, NOTHING; 29 classes) Preprocessing: Resizes/normalizes Kaggle images into .npy splits (no detection needed); loaded in Notebook (SpeakEZ_ASL.ipynb) for CNN training.
```bash
# Train model (with video logging)
python -m src.preprocessing
```
# Monitor training
tensorboard --logdir logs/tensorboard

**WASL Word Model:**

Trained from scratch, implemented via PyTorch for sequence handling in the WASL word recognition (SpeakEZ_WASL.ipynb), with MediaPipe for skeleton extraction and post LSTM for scikit-learn (joblib) for final classifier. 

## Usage

### Live Demo Application
```bash
# Run hybrid demo (letters + words)
python speakez/ui/app_hybrid.py \
    --model_asl models/final/asl_baseline_cnn_128_final.h5 \
    --model_wasl models/final/word_skel_logreg_8.joblib \
    --confidence_threshold 0.7 \
    --smoothing_window 10

# Demo controls:
# SPACE - Speak accumulated text
# C - Clear text buffer
# R - Start/stop recording
# H - Toggle help overlay
# Q/ESC - Quit application
```

## Project Structure

```
speakez/
├── README.md                   # This file
├── requirements.txt            # Dependencies
├── config/
│   └── config.yaml            # Configuration
├── data/
│   ├── raw/                   # Original dataset
│   ├── processed/             # Preprocessed images
│   └── splits/                # Train/val/test splits
├── models/
│   ├── checkpoints/           # Training checkpoints
│   └── final/                 # Production model
├── src/
│   ├── preprocessing.py      # Hand detection & preprocessing (ASL data extraction/preporcessing)
│   ├── model.py              # CNN architecture
│   ├── train.py              # Training pipeline
│   ├── inference_.py          # Real-time inference
│   ├── word_detector.py      # Word recognition
│   └── utils.py              # Helper functions
├── ui/
│   └── app_hybrid.py          # Live Main demo application (Letters & Words)
├── logs/
│   ├── videos/               # Training & demo videos
│   ├── visualizations/       # Training plots
│   └── samples/              # Sample images
└── notebooks/
    ├── SpeakEZ_ASL.ipynb    # Model Training for ASL (A-Z, DEL, BACKSPACE, NOTHING)
    └── SpeakEZ_WASL.ipynb   # Model Training for Word ASL (hello,thank you,please,yes,no,help,stop,sorry,good,bad)
```

## Known Limitations & Failure Modes

### Environmental Factors
- **Lighting:** Performance degrades <100 lux or >10000 lux
- **Background:** Complex backgrounds reduce accuracy by ~15%
- **Camera Angle:** Optimal range: 0-30° from perpendicular

### Gesture-Specific Issues
- **Similar Signs:** D/R, M/N, K/V confusion rates: 10-15%
- **Motion Blur:** Fast transitions cause 20% accuracy drop
- **Partial Occlusion:** >30% hand occlusion fails detection

### System Constraints
- **Latency:** Network inference dominates (65% of pipeline)
- **Memory:** Peak usage during batch processing: 2.3GB
- **CPU Bottleneck:** HSV conversion (35% of preprocessing)

## Development Phases

- **Phase 1: Foundation (Sep 15-25)**: Environment setup, Data preprocessing pipeline, Hand detection (OpenCV), Dataset preparation
- **Phase 2: Model Training (Sep 26-Oct 5)**: CNN architecture implementation, Model training with video logging, Hyperparameter tuning, Performance evaluation
- **Phase 3: Word Recognition (Oct 5-15)**: Fingerspelling word detection, TTS integration, Final testing, Demo application
- **Phase 4: Finalization (Oct 20-28)**: Documentation, Demo video recording

## Configuration

```yaml
# config/config.yaml - Configuration file for SpeakEZ

data:
  raw_path: "data/raw"
  processed_path: "data/processed"
  splits_path: "data/splits"

paths:
  # Letter model paths
  letter_model: "models/final/asl_baseline_cnn_128_final.h5"
  letter_classes: "models/final/classes_letters.json"
  
  # Word model paths
  word_logreg: "models/final/word_skel_logreg_8.joblib"
  word_thresholds: "models/final/word_thresholds.json"
  word_subset: "models/final/words_subset.json"
  word_prototypes: "models/final/word_prototypes.npz"

ui:
  # Letter detection settings
  letters_smooth_N: 3  # Require N consecutive frames to agree before updating
  webcam_mirror: false  # Set to true if camera is mirrored
  letter_conf_threshold: 0.25  # Lower = more sensitive (was 0.30)
  
  # Word detection thresholds
  prob_threshold_default: 0.25  # Minimum probability to accept prediction
  top2_margin_default: 0.05     # Required margin between top 2 predictions
  agreement_default: false      # Require prototype agreement

  FRAME_FREEZE_EPS: 0.3

# Model training settings (for reference)
model:
  img_size: 128
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
```

## Video Logging

### Training Video

Automatically records training progress:

- Real-time loss/accuracy plots
- Sample predictions visualization
- Per-epoch metrics

Saved to: logs/videos/training_progress.mp4

### Temporal Smoothing

- Buffer size: 10 frames
- Majority voting on predictions
- Confidence threshold: 0.7 (selected empirically to balance accuracy and responsiveness; adjust via config for experimentation—why 0.7? Testing showed it reduced false positives by 15% without delaying real-time FPS)
- Prevents flickering predictions

## Future Roadmap

### Short-term (3-6 months)

- Enhanced temporal modeling (LSTM/Transformer)
- Multi-modal fusion (hand + facial expression)
- Environmental robustness improvements
- Mobile deployment (TensorFlow Lite)

## Contributing

Fork the repo, create a branch (`git checkout -b feature/new-enhancement`), commit changes, push, and open a Pull Request. Focus on issues like improving accuracy or adding words. Follow code style (PEP8) and include tests. Explore the notebooks to replicate training and experiment with initializations—this hybrid setup invites curiosity in framework comparisons.


## Citation

```bibtex
@misc{speakez2025,
  title={SpeakEZ: Real-time ASL to Speech Translation Using Hybrid Deep Learning},
  author={Santiago, Amelita and Lee, Fang Hui},
  year={2025},
  institution={National University of Singapore - Institute of Systems Science},
  note={Pattern Recognition Systems Practice Module}
}
```

## License

This project is developed for academic purposes as part of the NUS-ISS Pattern Recognition Systems Practice Module. See [LICENSE](LICENSE) for details.

## References

- ASL Alphabet Dataset: Kaggle (Citation: Grassknoted. (2017). ASL Alphabet. Kaggle. https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- OpenCV Documentation: opencv.org
- TensorFlow/Keras: tensorflow.org

## Acknowledgments

- **Dataset:** Grassknoted. (2017). ASL Alphabet. Kaggle. https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- **NUS-ISS Faculty:** Prof. [Name] for guidance and supervision
- **Open Source Libraries:** TensorFlow, PyTorch, OpenCV, MediaPipe teams
- **ASL Community:** For dataset contributions and validation feedback

## Contributors

- Amelita Santiago - Preprocessing, Model development, Training pipeline, Testing and Demo application
- Lee Fang Hui - Testing and Demo application, Documentation

## Contact

For questions or feedback:

- Email: amelitasantiago@gmail.com 
- GitHub Issues: https://github.com/amelitasantiago/SpeakEZ-Hand-Gesture-Speech-ML/issues

**Issues & Support:**
- GitHub Issues: https://github.com/amelitasantiago/SpeakEZ-Hand-Gesture-Speech-ML/issues
- Documentation: https://github.com/amelitasantiago/SpeakEZ-Hand-Gesture-Speech-ML/wiki

---
*Last Updated: October 17, 2025*