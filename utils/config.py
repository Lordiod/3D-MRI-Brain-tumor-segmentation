"""
Configuration settings for the brain tumor segmentation application.
"""

import os


class AppConfig:
    """Application configuration settings."""
    
    # Image processing settings
    IMG_SIZE = 64
    VOLUME_SLICES = 50
    VOLUME_START_AT = 22
    
    # Visualization settings
    DEFAULT_THRESHOLD = 0.4
    DEFAULT_SLICE = 25
    
    # Segmentation classes
    SEGMENT_CLASSES = {
        0: 'NOT tumor',
        1: 'NECROTIC/CORE', 
        2: 'EDEMA',
        3: 'ENHANCING'
    }
    
    # UI settings
    WINDOW_WIDTH = 1000
    WINDOW_HEIGHT = 700
    
    # Professional dark theme color scheme
    COLORS = {
        'primary': "#3B82F6",      # Blue for primary buttons
        'primary_dark': "#2563EB", # Darker blue for hover
        'secondary': "#6B7280",    # Medium gray
        'success': "#10B981",      # Green for success
        'warning': "#F59E0B",      # Amber for warnings
        'error': "#EF4444",        # Red for errors
        'background': "#111827",   # Very dark background
        'card_bg': "#1F2937",      # Dark card background
        'text_primary': "#F9FAFB", # Light text for primary
        'text_secondary': "#9CA3AF", # Medium gray for secondary text
        'border': "#374151"        # Dark border color
    }
    
    # Typography
    FONTS = {
        'heading': ("Segoe UI", 24, "bold"),
        'subheading': ("Segoe UI", 14, "bold"),
        'body': ("Segoe UI", 11),
        'caption': ("Segoe UI", 10),
        'button': ("Segoe UI", 11, "bold")
    }
    VISUALIZATION_WIDTH = 1400
    VISUALIZATION_HEIGHT = 900
    
    # Model settings
    DEFAULT_MODEL_FILENAME = "model_x1_1.h5"
    
    # File extensions
    SUPPORTED_EXTENSIONS = ['.nii', '.nii.gz']
    
    @classmethod
    def get_model_path(cls, custom_path=None):
        """
        Get the model file path.
        
        Args:
            custom_path (str, optional): Custom model path
            
        Returns:
            str: Model file path
        """
        if custom_path and os.path.exists(custom_path):
            return custom_path
        
        # Try current directory
        current_dir_path = os.path.join(os.getcwd(), cls.DEFAULT_MODEL_FILENAME)
        if os.path.exists(current_dir_path):
            return current_dir_path
        
        # Try parent directory (for compatibility with original structure)
        parent_dir_path = os.path.join(os.path.dirname(os.getcwd()), cls.DEFAULT_MODEL_FILENAME)
        if os.path.exists(parent_dir_path):
            return parent_dir_path
        
        # Return default path (may not exist)
        return current_dir_path
    
    @classmethod
    def get_class_colors(cls):
        """Get color mapping for segmentation classes."""
        return {
            0: 'black',     # Background/Not tumor
            1: 'red',       # Necrotic/Core
            2: 'green',     # Edema
            3: 'blue'       # Enhancing
        }
    
    @classmethod
    def get_class_colormaps(cls):
        """Get colormap names for individual class visualization."""
        return {
            1: 'Reds',      # Necrotic/Core
            2: 'Greens',    # Edema  
            3: 'Blues'      # Enhancing
        }
