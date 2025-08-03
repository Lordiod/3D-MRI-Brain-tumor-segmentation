"""
Main application window and UI components.
"""

import os
import sys
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox

from models.brain_tumor_model import BrainTumorModel
from utils.data_handler import ImageLoader, DataValidator
from utils.config import AppConfig
from ui.visualization_window import VisualizationWindow


class BrainTumorSegmentationApp:
    """Main application class for brain tumor segmentation."""
    
    def __init__(self):
        # Initialize UI
        self.root = ctk.CTk()
        self.root.title("3D MRI Brain Tumor Segmentation")
        self.root.geometry(f"{AppConfig.WINDOW_WIDTH}x{AppConfig.WINDOW_HEIGHT}")
        self.root.resizable(width=False, height=False)
        
        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Initialize components
        self.model = BrainTumorModel()
        self.image_loader = ImageLoader()
        self.data_validator = DataValidator()
        
        # Data variables
        self.flair_data = None
        self.t1ce_data = None
        self.predictions = None
        self.current_slice = AppConfig.DEFAULT_SLICE
        self.threshold = AppConfig.DEFAULT_THRESHOLD
        
        # UI components
        self.viz_window = VisualizationWindow(self)
        
        # Initialize UI and load model
        self.create_widgets()
        self.load_model()
        self.reset_data()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        """Create and layout UI widgets."""
        # Main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(
            self.main_frame,
            text="3D MRI Brain Tumor Segmentation",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=10)
        
        # Control frame
        control_frame = ctk.CTkFrame(self.main_frame)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        # File operations frame
        file_frame = ctk.CTkFrame(control_frame)
        file_frame.pack(fill="x", padx=5, pady=5)
        
        # Buttons
        self.load_flair_btn = ctk.CTkButton(
            file_frame,
            text="Load FLAIR Image",
            command=self.load_flair_image,
            width=120
        )
        self.load_flair_btn.pack(side="left", padx=5)
        
        self.load_t1ce_btn = ctk.CTkButton(
            file_frame,
            text="Load T1CE Image", 
            command=self.load_t1ce_image,
            width=120
        )
        self.load_t1ce_btn.pack(side="left", padx=5)
        
        self.predict_btn = ctk.CTkButton(
            file_frame,
            text="Predict Segmentation",
            command=self.predict_segmentation,
            width=120,
            state="disabled"
        )
        self.predict_btn.pack(side="left", padx=5)
        
        self.reset_btn = ctk.CTkButton(
            file_frame,
            text="Reset",
            command=self.reset_data,
            width=120
        )
        self.reset_btn.pack(side="left", padx=5)
        
        self.visualize_btn = ctk.CTkButton(
            file_frame,
            text="Open Visualization",
            command=self.open_visualization,
            width=120,
            state="disabled"
        )
        self.visualize_btn.pack(side="left", padx=5)
        
        # Status label
        self.status_label = ctk.CTkLabel(control_frame, text="Ready to load images...")
        self.status_label.pack(pady=5)
        
        # Info frame
        self.info_frame = ctk.CTkFrame(self.main_frame)
        self.info_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Instructions
        instructions = ctk.CTkLabel(
            self.info_frame,
            text="Instructions:\n\n"
                 "1. Load FLAIR and T1CE NIfTI images\n"
                 "2. Click 'Predict Segmentation'\n"
                 "3. Open results window\n"
                 "4. Adjust sliders in results\n"
                 "5. Use 'Reset' to start over",
            font=ctk.CTkFont(size=12),
            justify="left"
        )
        instructions.pack(pady=10)
        
        self.info_label = ctk.CTkLabel(
            self.info_frame,
            text="Ready to load images...",
            font=ctk.CTkFont(size=10)
        )
        self.info_label.pack(pady=5)
    
    def load_model(self):
        """Load the brain tumor segmentation model."""
        try:
            model_path = AppConfig.get_model_path()
            
            if self.model.load_model(model_path):
                self.status_label.configure(text="Model loaded successfully!")
                self.info_label.configure(text="Model loaded successfully! Ready to load images.")
            else:
                self.status_label.configure(text="Model file not found. Please ensure model_x1_1.h5 is available.")
                self.info_label.configure(text="Model file not found! Please ensure model_x1_1.h5 is available.")
                
            self.check_ready_to_predict()
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            self.status_label.configure(text=error_msg)
            self.info_label.configure(text=error_msg)
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def load_flair_image(self):
        """Load FLAIR NIfTI image."""
        file_path = filedialog.askopenfilename(
            title="Select FLAIR NIfTI file",
            filetypes=[("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.flair_data = self.image_loader.load_nifti(file_path)
                self.status_label.configure(text=f"FLAIR loaded: {os.path.basename(file_path)}")
                self.info_label.configure(text=f"FLAIR loaded: {os.path.basename(file_path)}")
                self.check_ready_to_predict()
                
            except Exception as e:
                print(f"Error loading FLAIR: {str(e)}")
                messagebox.showerror("Error", f"Failed to load FLAIR image: {str(e)}")
    
    def load_t1ce_image(self):
        """Load T1CE NIfTI image."""
        file_path = filedialog.askopenfilename(
            title="Select T1CE NIfTI file",
            filetypes=[("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.t1ce_data = self.image_loader.load_nifti(file_path)
                self.status_label.configure(text=f"T1CE loaded: {os.path.basename(file_path)}")
                self.info_label.configure(text=f"T1CE loaded: {os.path.basename(file_path)}")
                self.check_ready_to_predict()
                
            except Exception as e:
                print(f"Error loading T1CE: {str(e)}")
                messagebox.showerror("Error", f"Failed to load T1CE image: {str(e)}")
    
    def reset_data(self):
        """Reset all data variables to clear memory."""
        self.flair_data = None
        self.t1ce_data = None
        self.predictions = None
        self.predict_btn.configure(state="disabled")
        self.visualize_btn.configure(state="disabled")
        
        if hasattr(self, 'viz_window') and self.viz_window.window is not None:
            self.viz_window.window.destroy()
        
        self.status_label.configure(text="Data reset. Ready to load images...")
        self.info_label.configure(text="Data reset. Ready to load new images...")
        print("All data and visualization reset")
        self.check_ready_to_predict()
    
    def check_ready_to_predict(self):
        """Check if prediction requirements are met and update UI."""
        is_valid, message = self.data_validator.validate_prediction_requirements(
            self.flair_data, self.t1ce_data, self.model
        )
        
        print(f"Checking readiness: {message}")
        
        if is_valid:
            self.predict_btn.configure(state="normal")
            print("Predict button enabled")
        else:
            self.predict_btn.configure(state="disabled")
            print("Predict button disabled")
        
        # Enable visualization button if we have data to visualize
        viz_valid, _ = self.data_validator.validate_visualization_data(
            self.flair_data, self.t1ce_data, self.predictions
        )
        
        if viz_valid:
            self.visualize_btn.configure(state="normal")
        else:
            self.visualize_btn.configure(state="disabled")
    
    def open_visualization(self):
        """Open the visualization window manually."""
        viz_valid, error_msg = self.data_validator.validate_visualization_data(
            self.flair_data, self.t1ce_data, self.predictions
        )
        
        if viz_valid:
            self.viz_window.create_window()
        else:
            messagebox.showwarning("Warning", f"Cannot open visualization: {error_msg}")
    
    def predict_segmentation(self):
        """Predict tumor segmentation."""
        is_valid, error_msg = self.data_validator.validate_prediction_requirements(
            self.flair_data, self.t1ce_data, self.model
        )
        
        if not is_valid:
            messagebox.showwarning("Warning", error_msg)
            return
        
        # Disable button during prediction
        self.predict_btn.configure(state="disabled", text="Predicting...")
        self.status_label.configure(text="Running prediction...")
        
        # Run prediction in separate thread
        threading.Thread(target=self._run_prediction, daemon=True).start()
    
    def _run_prediction(self):
        """Run the actual prediction in background thread."""
        try:
            # Make prediction using the model
            self.predictions = self.model.predict(self.flair_data, self.t1ce_data)
            
            # Update GUI in main thread
            self.root.after(0, self._prediction_complete)
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self._prediction_error(str(e)))
    
    def _prediction_complete(self):
        """Handle prediction completion in main thread."""
        self.predict_btn.configure(state="normal", text="Predict Segmentation")
        self.visualize_btn.configure(state="normal")
        self.status_label.configure(text="Prediction completed successfully!")
        self.info_label.configure(text="Prediction completed! Opening visualization window...")
        
        if self.predictions is not None:
            # Get prediction summary
            summary = self.model.get_prediction_summary(self.predictions, self.threshold)
            print(f"Opening visualization with predictions shape={summary['shape']}")
            
            # Check if predictions are above threshold
            above_threshold = any(summary['above_threshold'].values())
            if not above_threshold:
                self.info_label.configure(text="Warning: Predictions are below threshold. Try lowering the threshold.")
                messagebox.showwarning(
                    "Warning", 
                    "Predictions are below the current threshold. Try lowering the threshold using the slider."
                )
            
            self.viz_window.create_window()
        else:
            print("No predictions available to visualize")
            self.status_label.configure(text="Error: No predictions generated")
            self.info_label.configure(text="Error: No predictions generated")
            messagebox.showerror("Error", "No predictions generated. Please try again.")
    
    def _prediction_error(self, error_msg):
        """Handle prediction error in main thread."""
        self.predict_btn.configure(state="normal", text="Predict Segmentation")
        self.status_label.configure(text=f"Prediction failed: {error_msg}")
        self.info_label.configure(text=f"Prediction failed: {error_msg}")
        messagebox.showerror("Prediction Error", f"Failed to run prediction: {error_msg}")
    
    def update_slice(self, value):
        """Update the current slice."""
        self.current_slice = int(value)
        if hasattr(self.viz_window, 'window') and self.viz_window.window is not None:
            self.viz_window.slice_var.set(self.current_slice)
            self.viz_window.update_display()
    
    def update_threshold(self, value):
        """Update the prediction threshold."""
        self.threshold = float(value)
        if hasattr(self.viz_window, 'window') and self.viz_window.window is not None:
            self.viz_window.threshold_var.set(self.threshold)
            self.viz_window.update_display()
    
    def on_closing(self):
        """Handle window close event to exit application."""
        if hasattr(self, 'viz_window') and self.viz_window.window is not None:
            self.viz_window.window.destroy()
        self.root.destroy()
        sys.exit(0)
    
    def run(self):
        """Start the application."""
        self.root.mainloop()
