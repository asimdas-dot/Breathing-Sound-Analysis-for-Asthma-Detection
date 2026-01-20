#!/usr/bin/env python3
# app.py - Breathing Sound Analysis API
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Config
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============= ENDPOINTS =============

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'message': 'Backend API is running',
        'version': '1.0'
    }), 200

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'API is working!',
        'endpoints': [
            'GET /health',
            'GET /test',
            'POST /predict',
            'POST /analyze-xray'
        ]
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Spirometry prediction"""
    try:
        # Debug info for incoming request
        app.logger.info(f"DEBUG /predict - Content-Type: {request.content_type}, is_json: {request.is_json}")

        # Attempt to parse JSON safely (silent=True avoids raising), otherwise use form data
        data = request.get_json(silent=True)
        if data is None:
            data = request.form.to_dict()
        
        # Validate input
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        name = data.get('name', 'Patient')
        age = data.get('age', 0)
        
        # Extract symptom data from FormData
        symptoms = {
            'tiredness': data.get('Tiredness', 0),
            'dry_cough': data.get('Dry-Cough', 0),
            'difficulty_breathing': data.get('Difficulty-in-Breathing', 0),
            'sore_throat': data.get('Sore-Throat', 0),
            'pains': data.get('Pains', 0),
            'nasal_congestion': data.get('Nasal-Congestion', 0),
            'runny_nose': data.get('Runny-Nose', 0),
        }
        
        # Mock prediction (replace with actual model)
        return jsonify({
            'status': 'success',
            'patient_name': name,
            'age': age,
            'severity_label': 'Mild',
            'confidence': 92.5,
            'risk_level': 'Low',
            'recommendation': 'Regular monitoring advised',
            'symptoms': symptoms
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-xray', methods=['POST'])
def analyze_xray():
    """X-ray analysis"""
    try:
        if 'xray_image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['xray_image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Mock prediction
        return jsonify({
            'status': 'success',
            'prediction_label': 'Normal',
            'confidence': 88.5,
            'risk_level': 'Low',
            'file': filename
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Model information"""
    return jsonify({
        'status': 'success',
        'models': {
            'spirometry': {
                'type': 'RandomForest',
                'accuracy': '92%',
                'features': 19
            },
            'xray': {
                'type': 'CNN',
                'classes': 3,
                'input': '224x224x3'
            }
        }
    }), 200

# ============= MAIN =============

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🫁 BREATHING SOUND ANALYSIS - ASTHMA DETECTION")
    print("="*60)
    print("\n✅ Backend API Starting...")
    print("📍 Server: http://localhost:5000")
    print("🔌 CORS: Enabled for Frontend")
    print("\n📊 Available Endpoints:")
    print("   • GET  /health      - Health check")
    print("   • GET  /test        - Test connection")
    print("   • POST /predict     - Spirometry prediction")
    print("   • POST /analyze-xray - X-ray analysis")
    print("   • GET  /model-info  - Model details")
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
