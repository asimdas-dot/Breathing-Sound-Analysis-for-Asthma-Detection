"""
Configuration File for Breathing Sound Analysis System
سانس کی آواز تجزیہ کاری کے لیے ترتیبات
"""

# ============================================================
# SERVER CONFIGURATION
# ============================================================

# Flask Server
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True
FLASK_ENV = 'development'  # development or production

# CORS Settings
CORS_ALLOWED_ORIGINS = ['*']  # Change to specific domains in production

# ============================================================
# FILE UPLOAD CONFIGURATION
# ============================================================

# Upload folder path
UPLOAD_FOLDER = 'uploads'

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Max file size (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# Cleanup old uploads after (days)
UPLOAD_CLEANUP_DAYS = 30

# ============================================================
# SPIROMETRY MODEL CONFIGURATION
# ============================================================

# Spirometry CSV path
SPIROMETRY_CSV_PATH = '../input/processed-data.csv'

# Model parameters
SPIROMETRY_CONFIG = {
    'n_estimators': 150,      # Number of trees
    'max_depth': 20,          # Maximum tree depth
    'random_state': 42,
    'test_size': 0.2,         # 80-20 split
    'stratify': True
}

# Model file path
SPIROMETRY_MODEL_PATH = 'spirometry_model.pkl'

# ============================================================
# X-RAY CNN CONFIGURATION
# ============================================================

# Image dimensions
CNN_IMG_HEIGHT = 224
CNN_IMG_WIDTH = 224
CNN_CHANNELS = 3

# Model type: 'custom', 'mobilenet', or 'resnet'
CNN_MODEL_TYPE = 'custom'

# Training parameters
CNN_CONFIG = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'early_stopping_patience': 10
}

# Transfer Learning parameters (for MobileNetV2/ResNet50)
TRANSFER_LEARNING_CONFIG = {
    'epochs': 30,
    'batch_size': 32,
    'learning_rate': 0.0001,  # Lower for fine-tuning
    'unfreeze_layers': False   # Set True for fine-tuning
}

# Model file paths
CNN_MODEL_PATH = 'xray_cnn_model.keras'
CNN_METADATA_PATH = 'xray_model_metadata.pkl'

# ============================================================
# DATA AUGMENTATION CONFIGURATION
# ============================================================

DATA_AUGMENTATION = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': False,
    'fill_mode': 'nearest'
}

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = 'INFO'

# Log file path
LOG_FILE = 'app.log'

# Log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================
# DATABASE CONFIGURATION (if needed)
# ============================================================

# Database type: sqlite, mysql, postgresql
DATABASE_TYPE = 'sqlite'
DATABASE_NAME = 'asthma_detection.db'

# ============================================================
# SECURITY CONFIGURATION
# ============================================================

# API Key (set in production)
API_KEY = 'your-secret-key-here'

# JWT Secret
JWT_SECRET = 'your-jwt-secret-here'

# Password salt
PASSWORD_SALT_ROUNDS = 10

# ============================================================
# THRESHOLDS & PARAMETERS
# ============================================================

# Confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.7

# High risk threshold
HIGH_RISK_THRESHOLD = 0.85

# Asthma severity classes
ASTHMA_CLASSES = {
    'Normal': 0,
    'Asthma_Detected': 1,
    'Severe': 2
}

# Severity names
SEVERITY_NAMES = {
    'Mild': '🟢 Mild - Low Risk',
    'Moderate': '🟡 Moderate - Medium Risk',
    'None': '✅ None - No Asthma'
}

# ============================================================
# EMAIL CONFIGURATION (for notifications)
# ============================================================

SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SENDER_EMAIL = 'your-email@gmail.com'
SENDER_PASSWORD = 'your-app-password'

# ============================================================
# CACHE CONFIGURATION
# ============================================================

# Cache type: simple, redis, memcached
CACHE_TYPE = 'simple'

# Cache timeout (seconds)
CACHE_TIMEOUT = 300

# ============================================================
# MONITORING & ALERTING
# ============================================================

# Enable monitoring
ENABLE_MONITORING = True

# Prediction logging
LOG_PREDICTIONS = True

# Alert email on high-risk cases
ALERT_ON_HIGH_RISK = True

# Alert email recipients
ALERT_EMAIL_RECIPIENTS = ['doctor@hospital.com']

# ============================================================
# FEATURE IMPORTANCE THRESHOLDS
# ============================================================

# Minimum importance for feature display
MIN_FEATURE_IMPORTANCE = 0.01

# Top N features to show
TOP_N_FEATURES = 15

# ============================================================
# MODEL EVALUATION METRICS
# ============================================================

# Metrics to track
METRICS_TO_TRACK = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'auc_roc'
]

# ============================================================
# API RATE LIMITING
# ============================================================

# Enable rate limiting
ENABLE_RATE_LIMITING = True

# Requests per minute
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_PERIOD = 60

# ============================================================
# DEVELOPMENT MODE SETTINGS
# ============================================================

if FLASK_ENV == 'development':
    DEBUG = True
    TESTING = False
    JSON_SORT_KEYS = False
    
elif FLASK_ENV == 'production':
    DEBUG = False
    TESTING = False
    JSON_SORT_KEYS = True

# ============================================================
# PATHS & DIRECTORIES
# ============================================================

import os

# Get the absolute path of the Backend directory
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# Parent directories
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
INPUT_DIR = os.path.join(PROJECT_DIR, 'input')
OUTPUT_DIR = os.path.join(BACKEND_DIR, 'output')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# PRINT CONFIGURATION (for verification)
# ============================================================

def print_config():
    """Print current configuration"""
    print("\n" + "="*60)
    print("BREATHING SOUND ANALYSIS - CONFIGURATION")
    print("="*60)
    print(f"Server: {FLASK_HOST}:{FLASK_PORT}")
    print(f"Environment: {FLASK_ENV}")
    print(f"Debug Mode: {DEBUG}")
    print(f"Upload Folder: {UPLOAD_FOLDER}")
    print(f"Max File Size: {MAX_FILE_SIZE / (1024*1024):.0f}MB")
    print(f"Spirometry CSV: {SPIROMETRY_CSV_PATH}")
    print(f"CNN Model Type: {CNN_MODEL_TYPE}")
    print(f"Image Size: {CNN_IMG_HEIGHT}x{CNN_IMG_WIDTH}x{CNN_CHANNELS}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"High Risk Threshold: {HIGH_RISK_THRESHOLD}")
    print("="*60 + "\n")


if __name__ == '__main__':
    print_config()
