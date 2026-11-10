# 🎭 TruthVision: AI Deepfake Detection & Forensic System

**TruthVision** is an end-to-end Deep Learning and Computer Vision system designed to detect synthetically manipulated images and videos (deepfakes). It integrates transfer learning with **EfficientNet-B0**, **Grad-CAM explainability**, **temporal frame timeline analysis**, **frequency anomaly detection**, and a full-stack **Django 6 Web Dashboard**.

---

## 🌟 Key Features

1. **Multimodal Deepfake Detection**:
   - **Single Image Analysis**: Face ROI extraction, synthetic probability calculation, and artifact classification.
   - **Video Timeline Engine**: Frame-by-frame sampling, temporal inconsistency tracking, and flagged interval detection.
   - **Live Scanner Mode**: Real-time webcam streaming analysis with face bounding HUD.

2. **Explainable AI (Grad-CAM)**:
   - Visualizes exact facial regions (eyes, mouth, blending seams) that influenced the classification.
   - Interactive opacity slider to blend heatmap over face ROI.

3. **Forensic Intelligence Suite**:
   - Multi-factor risk calculation combining model probability, boundary saliency, and FFT spectral anomaly.
   - Downloadable / printable **Forensic Verification Certificate**.

4. **Model Comparison & Academic Benchmarks**:
   - Performance benchmarks comparing **EfficientNet-B0**, **XceptionNet**, **ResNet-50**, and **MobileNet-V3**.
   - Cross-dataset generalization matrix (**FaceForensics++** vs **Celeb-DF**).

---

## 📁 Project Structure

```text
truthvision/
│
├── dataset/                    # Dataset directory (train / val / test)
│   └── prepare_sample_data.py  # Synthetic benchmark data & test video generator
│
├── models/                     # Deep learning architectures
│   ├── __init__.py
│   └── detector.py             # EfficientNet / ResNet transfer learning models
│
├── preprocessing/              # Computer Vision preprocessing
│   ├── __init__.py
│   └── face_detector.py        # Margin-padded face extraction & tensor normalization
│
├── explainability/             # Explainable AI
│   ├── __init__.py
│   └── gradcam.py              # Grad-CAM heatmap generation & region analyzer
│
├── forensics/                  # Forensic analysis engine
│   ├── __init__.py
│   ├── risk_scorer.py          # Multi-factor authenticity risk calculator
│   └── report_generator.py     # Forensic certificate HTML/PDF generator
│
├── inference/                  # Prediction engines
│   ├── __init__.py
│   ├── image_predict.py        # Single image inference pipeline
│   └── video_engine.py         # Video temporal frame analysis pipeline
│
├── training/                   # Model training and evaluation
│   ├── __init__.py
│   ├── dataset.py              # PyTorch Dataset with data augmentations
│   ├── train.py                # CUDA mixed-precision training script
│   └── evaluate.py             # Metrics (F1, AUC, Confusion Matrix) evaluator
│
├── truthvision_project/        # Django Project configuration
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── webapp/                     # Django Application
│   ├── views.py                # Web & API views
│   ├── urls.py
│   ├── static/                 # CSS, JS, Chart.js assets
│   └── templates/              # Modern Dark Cyberpunk UI templates
│
├── checkpoints/                # Saved model weights (.pth)
├── requirements.txt
└── manage.py
```

---

## 🚀 Quick Start Guide

### 1. Generate Sample Benchmark Data
Generate test images and sample video immediately:
```bash
python dataset/prepare_sample_data.py
```

### 2. Train the Model (Optional / Custom Data)
Run mixed-precision GPU training on your dataset:
```bash
python -m training.train --train_dir dataset/train --val_dir dataset/validation --epochs 10 --batch_size 16
```

### 3. Evaluate Checkpoint
Compute Accuracy, Precision, Recall, F1, ROC-AUC, and confusion matrix:
```bash
python -m training.evaluate --checkpoint checkpoints/truthvision_efficientnet_b0_best.pth --test_dir dataset/test
```

### 4. Run the Django Web Dashboard
Apply database migrations and launch the local server:
```bash
python manage.py migrate
python manage.py runserver
```
Open **http://127.0.0.1:8000/** in your browser.
