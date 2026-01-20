"""
Test Script - Spirometry + X-ray CNN Models
Dono models ko test karne ke liye
"""

import sys
import os
import numpy as np
import pandas as pd

# Add Backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spirometry_classifier import SpirometryClassifier
from xray_cnn_analyzer import XrayClassifier
import tensorflow as tf
from tensorflow import keras


def test_spirometry_classifier():
    """Spirometry classifier ko test karo"""
    print("\n" + "="*70)
    print("🫁 TESTING SPIROMETRY CLASSIFIER (RandomForest)")
    print("="*70)
    
    try:
        classifier = SpirometryClassifier(csv_path='../input/processed-data.csv')
        
        # Load data
        print("\n1️⃣ Loading data...")
        classifier.load_data()
        
        # Prepare data
        print("\n2️⃣ Preparing data...")
        classifier.prepare_data(test_size=0.2)
        print(f"   Train samples: {len(classifier.X_train)}")
        print(f"   Test samples: {len(classifier.X_test)}")
        
        # Train model
        print("\n3️⃣ Training RandomForest model...")
        classifier.train_model(n_estimators=100, max_depth=15)
        
        # Evaluate
        print("\n4️⃣ Evaluating model...")
        metrics = classifier.evaluate_model()
        
        # Feature importance
        print("\n5️⃣ Top 10 important features:")
        classifier.feature_importance(top_n=10)
        
        # Save model
        print("\n6️⃣ Saving model...")
        classifier.save_model('spirometry_test_model.pkl')
        
        # Test prediction
        print("\n7️⃣ Testing single prediction...")
        sample_data = pd.DataFrame([{col: 1 for col in classifier.feature_names}])
        result = classifier.predict(sample_data)
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']*100:.2f}%")
        
        print("\n✅ Spirometry Classifier Test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Spirometry Classifier Test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_cnn():
    """Custom CNN ko test karo"""
    print("\n" + "="*70)
    print("🖼️ TESTING CUSTOM CNN")
    print("="*70)
    
    try:
        classifier = XrayClassifier(img_height=224, img_width=224)
        
        # Create model
        print("\n1️⃣ Creating Custom CNN architecture...")
        classifier.create_custom_cnn()
        
        # Compile
        print("\n2️⃣ Compiling model...")
        classifier.compile_model(learning_rate=0.001)
        
        # Create dummy data
        print("\n3️⃣ Creating dummy training data...")
        X_train = np.random.rand(50, 224, 224, 3).astype(np.float32)
        y_train = keras.utils.to_categorical(np.random.randint(0, 3, 50), 3)
        
        X_val = np.random.rand(10, 224, 224, 3).astype(np.float32)
        y_val = keras.utils.to_categorical(np.random.randint(0, 3, 10), 3)
        
        # Train
        print("\n4️⃣ Training model (5 epochs)...")
        classifier.train_from_arrays(X_train, y_train, X_val, y_val, epochs=5, batch_size=16)
        
        # Evaluate
        print("\n5️⃣ Evaluating model...")
        X_test = np.random.rand(10, 224, 224, 3).astype(np.float32)
        y_test = keras.utils.to_categorical(np.random.randint(0, 3, 10), 3)
        metrics = classifier.evaluate(X_test, y_test)
        
        # Save
        print("\n6️⃣ Saving model...")
        classifier.save_model('xray_custom_cnn_test.keras')
        
        print("\n✅ Custom CNN Test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Custom CNN Test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_mobilenet_transfer():
    """MobileNetV2 transfer learning ko test karo"""
    print("\n" + "="*70)
    print("🖼️ TESTING MobileNetV2 TRANSFER LEARNING")
    print("="*70)
    
    try:
        classifier = XrayClassifier(img_height=224, img_width=224)
        
        # Create model
        print("\n1️⃣ Creating MobileNetV2 Transfer Learning model...")
        classifier.create_mobilenet_transfer()
        
        # Compile
        print("\n2️⃣ Compiling model...")
        classifier.compile_model(learning_rate=0.0001)
        
        # Create dummy data
        print("\n3️⃣ Creating dummy training data...")
        X_train = np.random.rand(50, 224, 224, 3).astype(np.float32)
        y_train = keras.utils.to_categorical(np.random.randint(0, 3, 50), 3)
        
        X_val = np.random.rand(10, 224, 224, 3).astype(np.float32)
        y_val = keras.utils.to_categorical(np.random.randint(0, 3, 10), 3)
        
        # Train
        print("\n4️⃣ Training model (3 epochs - transfer learning is fast)...")
        classifier.train_from_arrays(X_train, y_train, X_val, y_val, epochs=3, batch_size=16)
        
        # Save
        print("\n5️⃣ Saving model...")
        classifier.save_model('xray_mobilenet_test.keras')
        
        print("\n✅ MobileNetV2 Transfer Learning Test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ MobileNetV2 Test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Saare tests run karo"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  BREATHING SOUND ANALYSIS - MODEL TESTING SUITE".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    results = {
        'Spirometry Classifier': False,
        'Custom CNN': False,
        'MobileNetV2 Transfer': False
    }
    
    # Test 1: Spirometry
    results['Spirometry Classifier'] = test_spirometry_classifier()
    
    # Test 2: Custom CNN
    results['Custom CNN'] = test_custom_cnn()
    
    # Test 3: MobileNetV2
    results['MobileNetV2 Transfer'] = test_mobilenet_transfer()
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print("\n" + "-"*70)
    print(f"Total: {total_passed}/{total_tests} tests passed")
    print("="*70)
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. Check logs above.")


if __name__ == '__main__':
    run_all_tests()
