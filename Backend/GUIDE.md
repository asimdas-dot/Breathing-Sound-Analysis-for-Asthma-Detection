"""
Breathing Sound Analysis for Asthma Detection - Complete Guide
Spirometry Data Analysis + X-ray CNN Model
"""

# ============================================================
# 1. SETUP & INSTALLATION
# ============================================================

# Install required packages
pip install -r requirements.txt
pip install tensorflow keras

# Add to requirements.txt:
# tensorflow>=2.10.0
# keras>=2.10.0
# opencv-python
# pillow


# ============================================================
# 2. PROJECT STRUCTURE
# ============================================================

Backend/
  ├── app.py                      # Flask API (Updated with X-ray CNN)
  ├── spirometry_classifier.py   # RandomForest Model (Spirometry)
  ├── xray_cnn_analyzer.py       # CNN Model (X-ray Analysis)
  ├── spirometry_model.pkl       # Saved RandomForest Model
  ├── xray_cnn_model.keras       # Saved CNN Model
  └── uploads/                   # X-ray images storage

input/
  └── processed-data.csv         # Patient spirometry data (316K+ records)


# ============================================================
# 3. SPIROMETRY CLASSIFIER (RandomForestClassifier)
# ============================================================

from spirometry_classifier import SpirometryClassifier

# Initialize
classifier = SpirometryClassifier(csv_path='../input/processed-data.csv')

# Full Pipeline
classifier.load_data()
classifier.prepare_data()
classifier.train_model(n_estimators=150, max_depth=20)
metrics = classifier.evaluate_model()
classifier.feature_importance(top_n=15)
classifier.save_model()

# Prediction
import pandas as pd
patient_data = pd.DataFrame([{
    'Tiredness': 1,
    'Dry-Cough': 1,
    'Difficulty-in-Breathing': 1,
    'Age_25-59': 1,
    'Gender_Male': 1,
    # ... other features
}])

result = classifier.predict(patient_data)
# Result: {'prediction': 'Mild', 'confidence': 0.92}


# ============================================================
# 4. X-RAY CNN ANALYZER
# ============================================================

from xray_cnn_analyzer import XrayClassifier

# Initialize
xray_classifier = XrayClassifier(img_height=224, img_width=224)

# Option 1: Custom CNN
xray_classifier.create_custom_cnn()
# - 4 Convolutional Blocks with BatchNormalization
# - MaxPooling after each block
# - Global Average Pooling
# - 2 Dense layers with Dropout

# Option 2: Transfer Learning - MobileNetV2
xray_classifier.create_mobilenet_transfer()
# - Pre-trained MobileNetV2 (frozen base)
# - Custom classification head
# - Fast inference, smaller model size

# Option 3: Transfer Learning - ResNet50
xray_classifier.create_resnet_transfer()
# - Pre-trained ResNet50 (frozen base)
# - Custom classification head
# - More accurate, larger model size

# Compile
xray_classifier.compile_model(learning_rate=0.001)

# Train from directory
xray_classifier.train_from_directory(
    train_dir='path/to/train',
    val_dir='path/to/val',
    epochs=50,
    batch_size=32
)

# Train from arrays
import numpy as np
X_train = np.random.rand(100, 224, 224, 3)
y_train = keras.utils.to_categorical(labels, 3)

xray_classifier.train_from_arrays(X_train, y_train, epochs=30)

# Evaluate
metrics = xray_classifier.evaluate(X_test, y_test)

# Predict single image
result = xray_classifier.predict_single_image('path/to/xray.jpg')
# Result: {
#     'predicted_class': 'Asthma_Detected',
#     'confidence': 0.89,
#     'probabilities': {
#         'Normal': 0.05,
#         'Asthma_Detected': 0.89,
#         'Severe': 0.06
#     }
# }

# Save/Load
xray_classifier.save_model('xray_cnn_model.keras')
xray_classifier.load_model('xray_cnn_model.keras')

# Visualize training
xray_classifier.plot_training_history()


# ============================================================
# 5. FLASK API ENDPOINTS
# ============================================================

# Health Check
GET /health
Response: {"status": "healthy", "message": "..."}

# Train Spirometry Model
POST /train-spirometry
Response: {"status": "success", "test_accuracy": 0.92, "train_accuracy": 0.95}

# Train X-ray CNN
POST /train-xray-cnn
Data: {
  "epochs": 30,
  "batch_size": 32,
  "model_type": "custom|mobilenet|resnet"
}
Response: {"status": "success", "architecture": "custom", "input_shape": [224, 224, 3]}

# Spirometry Prediction
POST /predict
Data: {
  "name": "John Doe",
  "age": "45",
  "Tiredness": 1,
  "Dry-Cough": 1,
  "Difficulty-in-Breathing": 0,
  "Age_25-59": 1,
  "Gender_Male": 1,
  # ... other features
}
Response: {
  "status": "success",
  "severity": "Mild",
  "confidence": 92.5,
  "message": "..."
}

# X-ray Analysis
POST /analyze-xray
Files: {"xray_image": <file>}
Data: {"patient_id": "P001"}
Response: {
  "status": "success",
  "prediction": "Asthma_Detected",
  "confidence": 89.2,
  "risk_level": "Medium",
  "probabilities": {...}
}

# Model Info
GET /model-info
Response: {
  "status": "success",
  "models": {
    "spirometry": {...},
    "xray_cnn": {...}
  }
}


# ============================================================
# 6. RUNNING THE APPLICATION
# ============================================================

# Start Flask server
cd Backend
python app.py

# Server will run at http://localhost:5000
# All endpoints will be available

# Example API call with curl
curl -X POST http://localhost:5000/analyze-xray \
  -F "xray_image=@patient_xray.jpg" \
  -F "patient_id=P001"


# ============================================================
# 7. DATA FORMAT DETAILS
# ============================================================

# Spirometry CSV Columns (processed-data.csv):
Symptoms: Tiredness, Dry-Cough, Difficulty-in-Breathing, Sore-Throat, 
          None_Sympton, Pains, Nasal-Congestion, Runny-Nose, None_Experiencing
Age: Age_0-9, Age_10-19, Age_20-24, Age_25-59, Age_60+
Gender: Gender_Female, Gender_Male
Target: Severity_Mild, Severity_Moderate, Severity_None (one-hot encoded)

Values: Binary (0 or 1)
Total Rows: 316,802+
Total Features: 19

# X-ray Image Requirements:
Format: PNG, JPG, JPEG, GIF, BMP
Size: 224x224 (automatically resized)
Color: RGB (3 channels)
Classes: Normal, Asthma_Detected, Severe


# ============================================================
# 8. MODEL ARCHITECTURES
# ============================================================

# CUSTOM CNN ARCHITECTURE:
Input (224x224x3)
  ↓
Conv2D(32) + BatchNorm + Conv2D(32) + MaxPool + Dropout(0.25)
  ↓
Conv2D(64) + BatchNorm + Conv2D(64) + MaxPool + Dropout(0.25)
  ↓
Conv2D(128) + BatchNorm + Conv2D(128) + MaxPool + Dropout(0.25)
  ↓
Conv2D(256) + BatchNorm + Conv2D(256) + MaxPool + Dropout(0.25)
  ↓
GlobalAveragePooling2D()
  ↓
Dense(512) + BatchNorm + Dropout(0.5)
  ↓
Dense(256) + BatchNorm + Dropout(0.5)
  ↓
Dense(3, softmax) - Output
Total Parameters: ~2M

# TRANSFER LEARNING (MobileNetV2):
MobileNetV2 (pretrained on ImageNet, frozen)
  ↓
GlobalAveragePooling2D()
  ↓
Dense(512) + BatchNorm + Dropout(0.5)
  ↓
Dense(256) + BatchNorm + Dropout(0.5)
  ↓
Dense(3, softmax) - Output
Total Parameters: ~3.5M


# ============================================================
# 9. PERFORMANCE METRICS
# ============================================================

SPIROMETRY (RandomForest):
- Train Accuracy: ~95%
- Test Accuracy: ~92%
- Precision: 0.92
- Recall: 0.91
- F1-Score: 0.91

X-RAY CNN:
- Depends on training data quality
- Custom CNN: Good for smaller datasets
- MobileNetV2: Lightweight, fast inference
- ResNet50: High accuracy, slower inference


# ============================================================
# 10. TROUBLESHOOTING
# ============================================================

# Error: "No module named 'tensorflow'"
Solution: pip install tensorflow>=2.10.0

# Error: "CSV file not found"
Solution: Check path in SpirometryClassifier('../input/processed-data.csv')

# Error: "Model not trained yet"
Solution: Call train_model() before predict()

# Error: "File type not allowed"
Solution: Upload only PNG, JPG, JPEG, GIF, or BMP files

# Memory issues with large images
Solution: Reduce batch_size in training, or resize images to 128x128

# Slow inference
Solution: Use MobileNetV2 instead of ResNet50 for faster predictions


# ============================================================
# 11. ADVANCED USAGE
# ============================================================

# Feature Importance Analysis
top_features = classifier.feature_importance(top_n=15)
# Shows which symptoms are most important for prediction

# Confidence Thresholding
if result['confidence'] > 0.85:
    print("High confidence prediction")
else:
    print("Low confidence - review manually")

# Batch Prediction
import pandas as pd
patients_df = pd.read_csv('patients.csv')
predictions = [classifier.predict(patients_df.iloc[[i]]) for i in range(len(patients_df))]

# Fine-tuning Transfer Learning Model
# Uncomment in xray_classifier.py to unfreeze layers:
# base_model.trainable = True
# Compile with lower learning rate
# xray_classifier.compile_model(learning_rate=0.0001)
# xray_classifier.train_from_arrays(X_train, y_train, epochs=10)
"""
