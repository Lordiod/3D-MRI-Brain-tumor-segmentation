"""
Model handling and prediction functionality for brain tumor segmentation.
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class BrainTumorModel:
    """Handle model loading and prediction operations."""
    
    def __init__(self, model_path=None):
        self.model = None
        self.IMG_SIZE = 64
        self.VOLUME_SLICES = 50
        self.VOLUME_START_AT = 22
        self.SEGMENT_CLASSES = {
            0: 'NOT tumor', 
            1: 'NECROTIC/CORE', 
            2: 'EDEMA', 
            3: 'ENHANCING'
        }
        
        if model_path:
            self.load_model(model_path)
    
    def dice_coef(self, y_true, y_pred, smooth=1.0):
        """Dice coefficient loss function."""
        class_num = 4
        for i in range(class_num):
            y_true_f = K.flatten(y_true[:,:,:,i])
            y_pred_f = K.flatten(y_pred[:,:,:,i])
            intersection = K.sum(y_true_f * y_pred_f)
            loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
            if i == 0:
                total_loss = loss
            else:
                total_loss = total_loss + loss
        total_loss = total_loss / class_num
        return total_loss
    
    def dice_coef_necrotic(self, y_true, y_pred, epsilon=1e-6):
        """Dice coefficient for necrotic class."""
        intersection = K.sum(K.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]))
        return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,1])) + K.sum(K.square(y_pred[:,:,:,1])) + epsilon)

    def dice_coef_edema(self, y_true, y_pred, epsilon=1e-6):
        """Dice coefficient for edema class."""
        intersection = K.sum(K.abs(y_true[:,:,:,2] * y_pred[:,:,:,2]))
        return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,2])) + K.sum(K.square(y_pred[:,:,:,2])) + epsilon)

    def dice_coef_enhancing(self, y_true, y_pred, epsilon=1e-6):
        """Dice coefficient for enhancing class."""
        intersection = K.sum(K.abs(y_true[:,:,:,3] * y_pred[:,:,:,3]))
        return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,3])) + K.sum(K.square(y_pred[:,:,:,3])) + epsilon)

    def precision(self, y_true, y_pred):
        """Calculate precision metric."""
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def sensitivity(self, y_true, y_pred):
        """Calculate sensitivity (recall) metric."""
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def specificity(self, y_true, y_pred):
        """Calculate specificity metric."""
        true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
        return true_negatives / (possible_negatives + K.epsilon())
    
    def load_model(self, model_path):
        """Load the trained model with custom objects."""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            custom_objects = {
                'accuracy': tf.keras.metrics.MeanIoU(num_classes=4),
                "dice_coef": self.dice_coef,
                "precision": self.precision,
                "sensitivity": self.sensitivity,
                "specificity": self.specificity,
                "dice_coef_necrotic": self.dice_coef_necrotic,
                "dice_coef_edema": self.dice_coef_edema,
                "dice_coef_enhancing": self.dice_coef_enhancing
            }
            
            self.model = keras.models.load_model(
                model_path, 
                custom_objects=custom_objects, 
                compile=False
            )
            print(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def is_loaded(self):
        """Check if model is loaded."""
        return self.model is not None
    
    def preprocess_data(self, flair_data, t1ce_data):
        """Preprocess FLAIR and T1CE data for prediction."""
        if flair_data is None or t1ce_data is None:
            raise ValueError("Both FLAIR and T1CE data are required")
        
        # Prepare input data
        X = np.zeros((self.VOLUME_SLICES, self.IMG_SIZE, self.IMG_SIZE, 2))
        
        for j in range(self.VOLUME_SLICES):
            slice_idx = j + self.VOLUME_START_AT
            if slice_idx < flair_data.shape[2] and slice_idx < t1ce_data.shape[2]:
                X[j, :, :, 0] = cv2.resize(flair_data[:, :, slice_idx], (self.IMG_SIZE, self.IMG_SIZE))
                X[j, :, :, 1] = cv2.resize(t1ce_data[:, :, slice_idx], (self.IMG_SIZE, self.IMG_SIZE))
        
        # Normalize input
        X = X / np.max(X) if np.max(X) > 0 else X
        
        print(f"Input prepared: shape={X.shape}, range={np.min(X):.2f} to {np.max(X):.2f}")
        return X
    
    def predict(self, flair_data, t1ce_data):
        """Make prediction on preprocessed data."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess data
        X = self.preprocess_data(flair_data, t1ce_data)
        
        # Make prediction
        predictions = self.model.predict(X, verbose=1)
        
        print(f"Prediction completed: shape={predictions.shape}, range={np.min(predictions):.2f} to {np.max(predictions):.2f}")
        print(f"Prediction mean per class: {np.mean(predictions, axis=(0,1,2))}")
        
        return predictions
    
    def get_prediction_summary(self, predictions, threshold=0.4):
        """Get summary statistics for predictions."""
        if predictions is None:
            return None
        
        max_probs = [np.max(predictions[:, :, :, i]) for i in range(4)]
        
        summary = {
            'shape': predictions.shape,
            'range': (np.min(predictions), np.max(predictions)),
            'max_probabilities': {
                self.SEGMENT_CLASSES[i]: max_probs[i] for i in range(4)
            },
            'above_threshold': {
                self.SEGMENT_CLASSES[i]: max_probs[i] > threshold for i in range(4)
            }
        }
        
        return summary
