"""
Spirometry Data Analysis using RandomForestClassifier
Yeh script patient data ko load karke train karta hai
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import pickle
import os

class SpirometryClassifier:
    def __init__(self, csv_path='../input/processed-data.csv'):
        """Initialize classifier with CSV file path"""
        self.csv_path = csv_path
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_column = 'Severity_Mild'  # Default target
        
    def load_data(self):
        """CSV file se data load karo"""
        print("📂 Data load ho raha hai...")
        self.data = pd.read_csv(self.csv_path)
        print(f"✓ Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        print(f"Columns: {list(self.data.columns)}")
        return self.data
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Features aur target ko separate karo"""
        print("\n📊 Data preparation...")
        
        # Target column ko identify karo
        # Agar Severity columns hain to unhe combine karo
        severity_cols = [col for col in self.data.columns if col.startswith('Severity_')]
        
        if severity_cols:
            # Multi-class classification: Severity ko target banao
            self.target_column = severity_cols
            # Target ko single column mein convert karo
            self.y = self.data[severity_cols].idxmax(axis=1)
            self.y = self.y.str.replace('Severity_', '')
            print(f"Target classes: {self.y.unique()}")
        else:
            raise ValueError("Severity columns nahi mile!")
        
        # Features ko prepare karo (saare columns except Severity)
        self.X = self.data.drop(columns=self.target_column)
        self.feature_names = self.X.columns.tolist()
        
        print(f"Features: {len(self.feature_names)} - {self.feature_names}")
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.y
        )
        
        print(f"✓ Training set: {self.X_train.shape[0]} samples")
        print(f"✓ Test set: {self.X_test.shape[0]} samples")
        
    def train_model(self, n_estimators=100, max_depth=15, random_state=42):
        """RandomForestClassifier ko train karo"""
        print("\n🤖 RandomForest Model training...")
        
        # Model initialize karo
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,  # Multi-processing
            verbose=1
        )
        
        # Model ko train karo
        self.model.fit(self.X_train, self.y_train)
        print("✓ Model training complete!")
        
    def evaluate_model(self):
        """Model ko evaluate karo"""
        print("\n📈 Model Evaluation...")
        
        # Predictions
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        
        # Metrics
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        
        print(f"\n📊 Accuracy:")
        print(f"  Train: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"  Test:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Detailed metrics
        print(f"\n📋 Classification Report (Test Set):")
        print(classification_report(self.y_test, y_pred_test))
        
        # Confusion Matrix
        print(f"\n🔀 Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred_test)
        print(cm)
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': classification_report(self.y_test, y_pred_test),
            'confusion_matrix': cm
        }
    
    def feature_importance(self, top_n=15):
        """Top features ke importance ko dikha"""
        print(f"\n🎯 Top {top_n} Important Features:")
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(importance_df.head(top_n).to_string(index=False))
        
        return importance_df.head(top_n)
    
    def save_model(self, model_path='spirometry_model.pkl'):
        """Model ko save karo"""
        print(f"\n💾 Model saving to {model_path}...")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'classes': self.model.classes_
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved!")
        
    def load_model(self, model_path='spirometry_model.pkl'):
        """Saved model ko load karo"""
        print(f"\n📂 Loading model from {model_path}...")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        
        print("✓ Model loaded!")
        
    def predict(self, patient_data):
        """Naye patient ka prediction karo"""
        if self.model is None:
            raise ValueError("Model train nahi ho gaya! Pehle train() call karo.")
        
        # Convert to DataFrame if needed
        if isinstance(patient_data, dict):
            patient_data = pd.DataFrame([patient_data])
        
        # Prediction
        prediction = self.model.predict(patient_data)[0]
        confidence = self.model.predict_proba(patient_data)[0].max()
        
        return {
            'prediction': prediction,
            'confidence': confidence
        }
    
    def full_pipeline(self):
        """Complete training pipeline"""
        print("="*60)
        print("🫁 SPIROMETRY DATA ANALYSIS - RANDOM FOREST CLASSIFIER")
        print("="*60)
        
        # Load
        self.load_data()
        
        # Prepare
        self.prepare_data()
        
        # Train
        self.train_model()
        
        # Evaluate
        metrics = self.evaluate_model()
        
        # Feature Importance
        self.feature_importance()
        
        # Save
        self.save_model()
        
        print("\n" + "="*60)
        print("✨ Pipeline Complete!")
        print("="*60)
        
        return metrics


def main():
    """Main function"""
    classifier = SpirometryClassifier(csv_path='../input/processed-data.csv')
    
    # Full pipeline run karo
    metrics = classifier.full_pipeline()
    
    # Example prediction
    print("\n\n📝 Example Prediction:")
    # Sample data - same structure as training data
    sample_patient = {col: 1 for col in classifier.feature_names}
    sample_patient[classifier.feature_names[0]] = 0
    
    result = classifier.predict(pd.DataFrame([sample_patient]))
    print(f"Patient Result: {result}")


if __name__ == '__main__':
    main()
