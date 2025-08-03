"""
Main entry point for the 3D MRI Brain Tumor Segmentation application.

This is the new modular version of the application, split into smaller components:
- models/: Contains the brain tumor model and prediction logic
- utils/: Contains data handling, configuration, and validation utilities  
- ui/: Contains the main window and visualization components

Usage:
    python main.py
"""

import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import BrainTumorSegmentationApp


def main():
    """Main function to run the application."""
    try:
        app = BrainTumorSegmentationApp()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Application error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
