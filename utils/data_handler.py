"""
Data handling utilities for brain tumor segmentation.
"""

import os
import numpy as np
import nibabel as nib


class ImageLoader:
    """Handle loading and validation of medical images."""
    
    def __init__(self):
        self.supported_extensions = ['.nii', '.nii.gz']
    
    def is_valid_file(self, file_path):
        """Check if file has valid extension."""
        if not file_path:
            return False
        return any(file_path.lower().endswith(ext) for ext in self.supported_extensions)
    
    def load_nifti(self, file_path):
        """
        Load NIfTI image file.
        
        Args:
            file_path (str): Path to the NIfTI file
            
        Returns:
            numpy.ndarray: Image data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not self.is_valid_file(file_path):
            raise ValueError(f"Unsupported file format. Supported: {self.supported_extensions}")
        
        try:
            image_data = nib.load(file_path).get_fdata()
            print(f"Loaded {os.path.basename(file_path)}: shape={image_data.shape}, dtype={image_data.dtype}")
            print(f"Data range: {np.min(image_data):.2f} to {np.max(image_data):.2f}")
            return image_data
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")
    
    def validate_image_compatibility(self, flair_data, t1ce_data):
        """
        Validate that FLAIR and T1CE images are compatible.
        
        Args:
            flair_data (numpy.ndarray): FLAIR image data
            t1ce_data (numpy.ndarray): T1CE image data
            
        Returns:
            bool: True if compatible
            
        Raises:
            ValueError: If images are incompatible
        """
        if flair_data is None or t1ce_data is None:
            raise ValueError("Both FLAIR and T1CE data must be provided")
        
        if flair_data.shape != t1ce_data.shape:
            raise ValueError(f"Image shape mismatch: FLAIR {flair_data.shape} vs T1CE {t1ce_data.shape}")
        
        if len(flair_data.shape) != 3:
            raise ValueError(f"Expected 3D images, got FLAIR shape: {flair_data.shape}")
        
        print(f"Images are compatible: shape={flair_data.shape}")
        return True


class DataValidator:
    """Validate data integrity and requirements."""
    
    @staticmethod
    def validate_prediction_requirements(flair_data, t1ce_data, model):
        """
        Validate that all requirements for prediction are met.
        
        Args:
            flair_data: FLAIR image data
            t1ce_data: T1CE image data
            model: Brain tumor model instance
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if model is None:
            return False, "Model is not loaded"
        
        if not model.is_loaded():
            return False, "Model is not properly loaded"
        
        if flair_data is None:
            return False, "FLAIR image is not loaded"
        
        if t1ce_data is None:
            return False, "T1CE image is not loaded"
        
        try:
            loader = ImageLoader()
            loader.validate_image_compatibility(flair_data, t1ce_data)
        except ValueError as e:
            return False, str(e)
        
        return True, "All requirements met"
    
    @staticmethod
    def validate_visualization_data(flair_data=None, t1ce_data=None, predictions=None):
        """
        Validate data for visualization.
        
        Args:
            flair_data: FLAIR image data (optional)
            t1ce_data: T1CE image data (optional)
            predictions: Prediction data (optional)
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if flair_data is None and t1ce_data is None and predictions is None:
            return False, "At least one data type must be provided for visualization"
        
        # Check if we have image data for background
        has_background = flair_data is not None or t1ce_data is not None
        
        if predictions is not None and not has_background:
            return False, "Background image (FLAIR or T1CE) required for prediction visualization"
        
        return True, "Visualization data is valid"
