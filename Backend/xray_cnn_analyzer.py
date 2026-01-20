"""
X-ray Image Analysis using TensorFlow/Keras CNN
Chest X-ray se asthma ke signs ko detect karne ke liye
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report, roc_auc_score
)
import pickle


class XrayClassifier:
    """X-ray images ko analyze karne ke liye CNN"""
    
    def __init__(self, img_height=224, img_width=224, channels=3):
        """Initialize CNN parameters"""
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.input_shape = (img_height, img_width, channels)
        self.model = None
        self.history = None
        self.class_names = ['Normal', 'Asthma_Detected', 'Severe']
        
    def create_custom_cnn(self):
        """Custom CNN architecture banao"""
        print("🏗️ Building Custom CNN Architecture...")
        
        self.model = models.Sequential([
            # Block 1
            layers.Input(shape=self.input_shape),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer (3 classes)
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        print("✓ Custom CNN created!")
        self.model.summary()
        return self.model
    
    def create_mobilenet_transfer(self):
        """Transfer Learning ke saath MobileNetV2 use karo"""
        print("🏗️ Building MobileNetV2 Transfer Learning Model...")
        
        # Pre-trained MobileNetV2 load karo
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Pehle ke layers ko freeze karo
        base_model.trainable = False
        
        # Custom head add karo
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        self.model = model
        print("✓ MobileNetV2 model created!")
        self.model.summary()
        return self.model
    
    def create_resnet_transfer(self):
        """Transfer Learning ke saath ResNet50 use karo"""
        print("🏗️ Building ResNet50 Transfer Learning Model...")
        
        # Pre-trained ResNet50 load karo
        base_model = ResNet50(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Pehle ke layers ko freeze karo
        base_model.trainable = False
        
        # Custom head add karo
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        self.model = model
        print("✓ ResNet50 model created!")
        self.model.summary()
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Model ko compile karo"""
        print("\n⚙️ Compiling model...")
        
        optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        print("✓ Model compiled!")
    
    def setup_data_augmentation(self):
        """Image augmentation setup"""
        print("\n📸 Setting up data augmentation...")
        
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        self.val_datagen = ImageDataGenerator(rescale=1./255)
        
        print("✓ Data augmentation ready!")
        
        return self.train_datagen, self.val_datagen
    
    def load_and_preprocess_image(self, img_path):
        """Single image ko load aur preprocess karo"""
        img = load_img(img_path, target_size=(self.img_height, self.img_width))
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        return np.expand_dims(img_array, axis=0)  # Batch dimension add karo
    
    def train_from_directory(self, train_dir, val_dir=None, epochs=50, batch_size=32):
        """Directory se images load karke train karo"""
        print("\n🎯 Training model from directory...")
        
        train_datagen, val_datagen = self.setup_data_augmentation()
        
        # Training data load karo
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        # Validation data load karo (agar available ho)
        val_generator = None
        if val_dir and os.path.exists(val_dir):
            val_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=(self.img_height, self.img_width),
                batch_size=batch_size,
                class_mode='categorical'
            )
        
        # Store class names
        self.class_names = list(train_generator.class_indices.keys())
        print(f"Classes: {self.class_names}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if val_generator else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if val_generator else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'xray_best_model.keras',
                monitor='val_loss' if val_generator else 'loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train karo
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✓ Training complete!")
        return self.history
    
    def train_from_arrays(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """Arrays se train karo"""
        print("\n🎯 Training model from arrays...")
        
        # Normalize images
        X_train = X_train / 255.0
        if X_val is not None:
            X_val = X_val / 255.0
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'xray_best_model.keras',
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train karo
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val) if X_val is not None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✓ Training complete!")
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Model ko test set par evaluate karo"""
        print("\n📊 Model Evaluation...")
        
        # Normalize
        X_test = X_test / 255.0
        
        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"\n✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"✅ Precision: {precision:.4f}")
        print(f"✅ Recall:    {recall:.4f}")
        print(f"✅ F1-Score:  {f1:.4f}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        print(f"\n🔀 Confusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def plot_training_history(self):
        """Training history ko visualize karo"""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in self.history.history:
            axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        if 'val_loss' in self.history.history:
            axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("✓ Training history saved to training_history.png")
        plt.close()
    
    def predict_single_image(self, img_path):
        """Single X-ray image ka prediction"""
        print(f"\n🔍 Predicting for: {img_path}")
        
        # Image load aur preprocess karo
        img_array = self.load_and_preprocess_image(img_path)
        
        # Prediction
        prediction = self.model.predict(img_array, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        
        result = {
            'image_path': img_path,
            'predicted_class': self.class_names[class_idx],
            'confidence': float(confidence),
            'probabilities': {
                self.class_names[i]: float(prediction[0][i]) 
                for i in range(len(self.class_names))
            }
        }
        
        print(f"Prediction: {result['predicted_class']}")
        print(f"Confidence: {confidence*100:.2f}%")
        print(f"All probabilities: {result['probabilities']}")
        
        return result
    
    def save_model(self, model_path='xray_cnn_model.keras'):
        """Model ko save karo"""
        print(f"\n💾 Saving model to {model_path}...")
        self.model.save(model_path)
        
        # Metadata save karo
        metadata = {
            'class_names': self.class_names,
            'img_height': self.img_height,
            'img_width': self.img_width,
            'channels': self.channels
        }
        
        with open('xray_model_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Model and metadata saved!")
    
    def load_model(self, model_path='xray_cnn_model.keras'):
        """Saved model ko load karo"""
        print(f"\n📂 Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        
        # Metadata load karo
        try:
            with open('xray_model_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            self.class_names = metadata['class_names']
        except:
            pass
        
        print("✓ Model loaded!")


def demo_custom_cnn():
    """Custom CNN demo"""
    print("="*70)
    print("🫁 CUSTOM CNN FOR X-RAY ANALYSIS")
    print("="*70)
    
    classifier = XrayClassifier(img_height=224, img_width=224)
    classifier.create_custom_cnn()
    classifier.compile_model(learning_rate=0.001)
    
    # Dummy data banao demo ke liye
    X_train = np.random.rand(100, 224, 224, 3)
    y_train = keras.utils.to_categorical(np.random.randint(0, 3, 100), 3)
    
    X_val = np.random.rand(20, 224, 224, 3)
    y_val = keras.utils.to_categorical(np.random.randint(0, 3, 20), 3)
    
    # Train karo
    classifier.train_from_arrays(X_train, y_train, X_val, y_val, epochs=5)
    
    # Evaluate karo
    X_test = np.random.rand(20, 224, 224, 3)
    y_test = keras.utils.to_categorical(np.random.randint(0, 3, 20), 3)
    classifier.evaluate(X_test, y_test)
    
    # Save karo
    classifier.save_model()
    
    print("\n" + "="*70)
    print("✨ Custom CNN Demo Complete!")
    print("="*70)


def demo_transfer_learning():
    """Transfer Learning (MobileNetV2) demo"""
    print("\n" + "="*70)
    print("🫁 TRANSFER LEARNING (MobileNetV2) FOR X-RAY ANALYSIS")
    print("="*70)
    
    classifier = XrayClassifier(img_height=224, img_width=224)
    classifier.create_mobilenet_transfer()
    classifier.compile_model(learning_rate=0.0001)
    
    # Dummy data banao demo ke liye
    X_train = np.random.rand(100, 224, 224, 3)
    y_train = keras.utils.to_categorical(np.random.randint(0, 3, 100), 3)
    
    X_val = np.random.rand(20, 224, 224, 3)
    y_val = keras.utils.to_categorical(np.random.randint(0, 3, 20), 3)
    
    # Train karo
    classifier.train_from_arrays(X_train, y_train, X_val, y_val, epochs=5)
    
    # Save karo
    classifier.save_model('xray_mobilenet_model.keras')
    
    print("\n" + "="*70)
    print("✨ Transfer Learning Demo Complete!")
    print("="*70)


if __name__ == '__main__':
    # Uncomment karo jo run karna hai
    demo_custom_cnn()
    # demo_transfer_learning()
