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
        
        # Set minimum and maximum window size to prevent resizing
        self.root.minsize(AppConfig.WINDOW_WIDTH, AppConfig.WINDOW_HEIGHT)
        self.root.maxsize(AppConfig.WINDOW_WIDTH, AppConfig.WINDOW_HEIGHT)
        
        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
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
        """Create and layout UI widgets with professional design."""
        # Configure main window
        self.root.configure(fg_color=AppConfig.COLORS['background'])
        
        # Main container with fixed size and proper padding
        self.main_container = ctk.CTkFrame(
            self.root,
            fg_color="transparent",
            width=AppConfig.WINDOW_WIDTH - 60,  # Account for padding
            height=AppConfig.WINDOW_HEIGHT - 50  # Account for padding
        )
        self.main_container.pack(fill="both", expand=True, padx=30, pady=25)
        self.main_container.pack_propagate(False)  # Prevent size changes
        
        # Header section
        self.create_header()
        
        # Content area with cards
        self.create_content_area()
        
        # Status bar
        self.create_status_bar()
    
    def create_header(self):
        """Create professional header section."""
        header_frame = ctk.CTkFrame(
            self.main_container,
            fg_color=AppConfig.COLORS['card_bg'],
            corner_radius=12,
            border_width=1,
            border_color=AppConfig.COLORS['border']
        )
        header_frame.pack(fill="x", pady=(0, 20))
        
        # Header content
        header_content = ctk.CTkFrame(header_frame, fg_color="transparent")
        header_content.pack(fill="x", padx=25, pady=20)
        
        # Title and subtitle
        title_label = ctk.CTkLabel(
            header_content,
            text="3D MRI Brain Tumor Segmentation",
            font=AppConfig.FONTS['heading'],
            text_color=AppConfig.COLORS['text_primary']
        )
        title_label.pack(anchor="w")
        
        subtitle_label = ctk.CTkLabel(
            header_content,
            text="AI-powered medical imaging analysis and visualization",
            font=AppConfig.FONTS['body'],
            text_color=AppConfig.COLORS['text_secondary']
        )
        subtitle_label.pack(anchor="w", pady=(5, 0))
    
    def create_content_area(self):
        """Create main content area with card-based layout."""
        # Content wrapper with fixed height
        content_wrapper = ctk.CTkFrame(
            self.main_container,
            fg_color="transparent",
            height=520  # Fixed height to prevent layout shifts
        )
        content_wrapper.pack(fill="both", expand=True)
        content_wrapper.pack_propagate(False)  # Prevent size changes
        
        # Left column - File operations
        left_column = ctk.CTkFrame(
            content_wrapper,
            fg_color="transparent",
            width=460  # Fixed width
        )
        left_column.pack(side="left", fill="y", padx=(0, 10))
        left_column.pack_propagate(False)  # Prevent size changes
        
        # File operations card
        self.create_file_operations_card(left_column)
        
        # Actions card
        self.create_actions_card(left_column)
        
        # Right column - Workflow and status
        right_column = ctk.CTkFrame(
            content_wrapper,
            fg_color="transparent",
            width=460  # Fixed width
        )
        right_column.pack(side="right", fill="both", expand=True, padx=(10, 0))
        right_column.pack_propagate(False)  # Prevent size changes
        
        # Workflow card
        self.create_workflow_card(right_column)
    
    def create_file_operations_card(self, parent):
        """Create file operations card."""
        card = ctk.CTkFrame(
            parent,
            fg_color=AppConfig.COLORS['card_bg'],
            corner_radius=12,
            border_width=1,
            border_color=AppConfig.COLORS['border'],
            height=230  # Fixed height
        )
        card.pack(fill="x", pady=(0, 15))
        card.pack_propagate(False)  # Prevent size changes
        
        # Card header
        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(20, 10))
        
        card_title = ctk.CTkLabel(
            header,
            text="📁 Load Medical Images",
            font=AppConfig.FONTS['subheading'],
            text_color=AppConfig.COLORS['text_primary']
        )
        card_title.pack(anchor="w")
        
        card_desc = ctk.CTkLabel(
            header,
            text="Select FLAIR and T1CE NIfTI files for analysis",
            font=AppConfig.FONTS['caption'],
            text_color=AppConfig.COLORS['text_secondary']
        )
        card_desc.pack(anchor="w", pady=(2, 0))
        
        # File buttons container
        buttons_frame = ctk.CTkFrame(card, fg_color="transparent")
        buttons_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        # FLAIR button with status
        flair_container = ctk.CTkFrame(buttons_frame, fg_color="transparent")
        flair_container.pack(fill="x", pady=(0, 10))
        
        self.load_flair_btn = ctk.CTkButton(
            flair_container,
            text="📋 Load FLAIR Image",
            command=self.load_flair_image,
            font=AppConfig.FONTS['button'],
            fg_color=AppConfig.COLORS['primary'],
            hover_color=AppConfig.COLORS['primary_dark'],
            corner_radius=8,
            height=45,
            width=300  # Fixed width to ensure consistency
        )
        self.load_flair_btn.pack(side="left", padx=(0, 10))
        
        self.flair_status = ctk.CTkLabel(
            flair_container,
            text="❌ Not loaded",
            font=AppConfig.FONTS['caption'],
            text_color=AppConfig.COLORS['text_secondary'],
            height=45,
            width=120
        )
        self.flair_status.pack(side="right")
        
        # T1CE button with status
        t1ce_container = ctk.CTkFrame(buttons_frame, fg_color="transparent")
        t1ce_container.pack(fill="x")
        
        self.load_t1ce_btn = ctk.CTkButton(
            t1ce_container,
            text="📋 Load T1CE Image",
            command=self.load_t1ce_image,
            font=AppConfig.FONTS['button'],
            fg_color=AppConfig.COLORS['primary'],
            hover_color=AppConfig.COLORS['primary_dark'],
            corner_radius=8,
            height=45,
            width=300  # Fixed width to match FLAIR button
        )
        self.load_t1ce_btn.pack(side="left", padx=(0, 10))
        
        self.t1ce_status = ctk.CTkLabel(
            t1ce_container,
            text="❌ Not loaded",
            font=AppConfig.FONTS['caption'],
            text_color=AppConfig.COLORS['text_secondary'],
            width=120
        )
        self.t1ce_status.pack(side="right")
    
    def create_actions_card(self, parent):
        """Create actions card."""
        card = ctk.CTkFrame(
            parent,
            fg_color=AppConfig.COLORS['card_bg'],
            corner_radius=12,
            border_width=1,
            border_color=AppConfig.COLORS['border'],
            height=160  # Fixed height
        )
        card.pack(fill="x")
        card.pack_propagate(False)  # Prevent size changes
        
        # Card header
        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(20, 10))
        
        card_title = ctk.CTkLabel(
            header,
            text="⚡ Actions",
            font=AppConfig.FONTS['subheading'],
            text_color=AppConfig.COLORS['text_primary']
        )
        card_title.pack(anchor="w")
        
        # Actions container
        actions_frame = ctk.CTkFrame(card, fg_color="transparent")
        actions_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        # Primary action button
        self.predict_btn = ctk.CTkButton(
            actions_frame,
            text="🧠 Predict Segmentation",
            command=self.predict_segmentation,
            font=AppConfig.FONTS['button'],
            fg_color=AppConfig.COLORS['success'],
            hover_color="#059669",  # Darker green for dark theme
            corner_radius=8,
            height=50,
            state="disabled"
        )
        self.predict_btn.pack(fill="x", pady=(0, 10))
        
        # Secondary actions row
        secondary_row = ctk.CTkFrame(actions_frame, fg_color="transparent")
        secondary_row.pack(fill="x")
        
        self.visualize_btn = ctk.CTkButton(
            secondary_row,
            text="📊 Visualize",
            command=self.open_visualization,
            font=AppConfig.FONTS['button'],
            fg_color=AppConfig.COLORS['secondary'],
            hover_color="#1F2937",  # Darker gray for dark theme
            corner_radius=8,
            height=40,
            state="disabled"
        )
        self.visualize_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.reset_btn = ctk.CTkButton(
            secondary_row,
            text="🔄 Reset",
            command=self.reset_data,
            font=AppConfig.FONTS['button'],
            fg_color=AppConfig.COLORS['warning'],
            hover_color="#D97706",  # Keep amber hover for visibility
            corner_radius=8,
            height=40
        )
        self.reset_btn.pack(side="right", fill="x", expand=True, padx=(5, 0))
    
    def create_workflow_card(self, parent):
        """Create workflow progress card."""
        card = ctk.CTkFrame(
            parent,
            fg_color=AppConfig.COLORS['card_bg'],
            corner_radius=12,
            border_width=1,
            border_color=AppConfig.COLORS['border'],
            height=520  # Fixed height to match content area
        )
        card.pack(fill="both", expand=True)
        card.pack_propagate(False)  # Prevent size changes
        
        # Card header
        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(20, 15))
        
        card_title = ctk.CTkLabel(
            header,
            text="📋 Workflow Progress",
            font=AppConfig.FONTS['subheading'],
            text_color=AppConfig.COLORS['text_primary']
        )
        card_title.pack(anchor="w")
        
        # Workflow steps
        workflow_frame = ctk.CTkFrame(
            card, 
            fg_color="transparent",
            height=420  # Fixed height for workflow content
        )
        workflow_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        workflow_frame.pack_propagate(False)  # Prevent size changes
        
        # Step indicators
        self.workflow_steps = []
        steps = [
            ("1", "Load FLAIR Image", "Upload your FLAIR NIfTI file"),
            ("2", "Load T1CE Image", "Upload your T1CE NIfTI file"),
            ("3", "Run Prediction", "AI analysis of brain images"),
            ("4", "View Results", "Explore segmentation results")
        ]
        
        for i, (num, title, desc) in enumerate(steps):
            step_frame = self.create_workflow_step(workflow_frame, num, title, desc, i == 0)
            step_frame.pack(fill="x", pady=(0, 15))
            self.workflow_steps.append(step_frame)
        
        # Info area
        self.info_area = ctk.CTkFrame(
            workflow_frame,
            fg_color=AppConfig.COLORS['background'],
            corner_radius=8,
            height=60  # Fixed height to prevent size changes
        )
        self.info_area.pack(fill="x", pady=(10, 0))
        self.info_area.pack_propagate(False)  # Prevent size changes
        
        self.info_label = ctk.CTkLabel(
            self.info_area,
            text="Ready to begin. Please load your medical images to start.",
            font=AppConfig.FONTS['body'],
            text_color=AppConfig.COLORS['text_secondary'],
            wraplength=400,  # Consistent text wrapping
            justify="left"
        )
        self.info_label.pack(padx=15, pady=12, expand=True)
    
    def create_workflow_step(self, parent, number, title, description, active=False):
        """Create a workflow step indicator."""
        step_frame = ctk.CTkFrame(parent, fg_color="transparent")
        
        # Step number circle
        circle_color = AppConfig.COLORS['primary'] if active else AppConfig.COLORS['border']
        text_color = "white" if active else AppConfig.COLORS['text_secondary']
        
        number_frame = ctk.CTkFrame(
            step_frame,
            width=30,
            height=30,
            fg_color=circle_color,
            corner_radius=15
        )
        number_frame.pack(side="left", padx=(0, 12))
        number_frame.pack_propagate(False)
        
        number_label = ctk.CTkLabel(
            number_frame,
            text=number,
            font=AppConfig.FONTS['caption'],
            text_color=text_color
        )
        number_label.pack(expand=True)
        
        # Step content
        content_frame = ctk.CTkFrame(step_frame, fg_color="transparent")
        content_frame.pack(side="left", fill="x", expand=True)
        
        title_label = ctk.CTkLabel(
            content_frame,
            text=title,
            font=AppConfig.FONTS['body'],
            text_color=AppConfig.COLORS['text_primary'] if active else AppConfig.COLORS['text_secondary']
        )
        title_label.pack(anchor="w")
        
        desc_label = ctk.CTkLabel(
            content_frame,
            text=description,
            font=AppConfig.FONTS['caption'],
            text_color=AppConfig.COLORS['text_secondary']
        )
        desc_label.pack(anchor="w")
        
        return step_frame
    
    def create_status_bar(self):
        """Create bottom status bar."""
        status_frame = ctk.CTkFrame(
            self.main_container,
            fg_color=AppConfig.COLORS['card_bg'],
            corner_radius=8,
            border_width=1,
            border_color=AppConfig.COLORS['border'],
            height=50  # Slightly increased height for better spacing
        )
        status_frame.pack(fill="x", pady=(20, 0))
        status_frame.pack_propagate(False)  # Prevent size changes
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="🔄 Ready - Load your medical images to begin analysis",
            font=AppConfig.FONTS['body'],
            text_color=AppConfig.COLORS['text_secondary'],
            wraplength=800  # Prevent text overflow
        )
        self.status_label.pack(pady=15, padx=20)  # Centered with padding
    
    def load_model(self):
        """Load the brain tumor segmentation model."""
        try:
            model_path = AppConfig.get_model_path()
            
            if self.model.load_model(model_path):
                self.status_label.configure(
                    text="✅ AI model loaded successfully - Ready for analysis",
                    text_color=AppConfig.COLORS['success']
                )
                self.info_label.configure(text="AI model loaded successfully! Ready to analyze medical images.")
            else:
                self.status_label.configure(
                    text="❌ Model file not found - Please ensure model_x1_1.h5 is available",
                    text_color=AppConfig.COLORS['error']
                )
                self.info_label.configure(text="❌ Model file not found! Please ensure model_x1_1.h5 is available in the project root.")
                
            self.check_ready_to_predict()
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            self.status_label.configure(text=f"❌ {error_msg}", text_color=AppConfig.COLORS['error'])
            self.info_label.configure(text=f"❌ {error_msg}")
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
                filename = os.path.basename(file_path)
                
                # Update UI elements
                self.flair_status.configure(text="✅ Loaded", text_color=AppConfig.COLORS['success'])
                self.status_label.configure(
                    text=f"✅ FLAIR loaded: {filename}",
                    text_color=AppConfig.COLORS['success']
                )
                self.info_label.configure(text=f"FLAIR image loaded successfully: {filename[:30]}..." if len(filename) > 30 else f"FLAIR image loaded successfully: {filename}")
                
                # Update workflow step
                self.update_workflow_step(0, True, "✅ FLAIR Loaded")
                
                self.check_ready_to_predict()
                
            except Exception as e:
                print(f"Error loading FLAIR: {str(e)}")
                self.flair_status.configure(text="❌ Error", text_color=AppConfig.COLORS['error'])
                self.status_label.configure(
                    text="❌ Failed to load FLAIR image",
                    text_color=AppConfig.COLORS['error']
                )
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
                filename = os.path.basename(file_path)
                
                # Update UI elements
                self.t1ce_status.configure(text="✅ Loaded", text_color=AppConfig.COLORS['success'])
                self.status_label.configure(
                    text=f"✅ T1CE loaded: {filename}",
                    text_color=AppConfig.COLORS['success']
                )
                self.info_label.configure(text=f"T1CE image loaded successfully: {filename[:30]}..." if len(filename) > 30 else f"T1CE image loaded successfully: {filename}")
                
                # Update workflow step
                self.update_workflow_step(1, True, "✅ T1CE Loaded")
                
                self.check_ready_to_predict()
                
            except Exception as e:
                print(f"Error loading T1CE: {str(e)}")
                self.t1ce_status.configure(text="❌ Error", text_color=AppConfig.COLORS['error'])
                self.status_label.configure(
                    text="❌ Failed to load T1CE image",
                    text_color=AppConfig.COLORS['error']
                )
                messagebox.showerror("Error", f"Failed to load T1CE image: {str(e)}")
    
    def reset_data(self):
        """Reset all data variables to clear memory."""
        self.flair_data = None
        self.t1ce_data = None
        self.predictions = None
        
        # Reset UI elements
        self.flair_status.configure(text="❌ Not loaded", text_color=AppConfig.COLORS['text_secondary'])
        self.t1ce_status.configure(text="❌ Not loaded", text_color=AppConfig.COLORS['text_secondary'])
        
        self.predict_btn.configure(state="disabled")
        self.visualize_btn.configure(state="disabled")
        
        # Reset workflow steps
        for i in range(4):
            self.update_workflow_step(i, False, ["Load FLAIR Image", "Load T1CE Image", "Run Prediction", "View Results"][i])
        
        # Activate first step
        self.update_workflow_step(0, True, "Load FLAIR Image")
        
        # Close visualization window if open
        if hasattr(self, 'viz_window') and self.viz_window.window is not None:
            self.viz_window.window.destroy()
        
        self.status_label.configure(
            text="🔄 Data reset - Ready to load new images",
            text_color=AppConfig.COLORS['text_secondary']
        )
        self.info_label.configure(text="All data cleared. Ready to load new medical images.")
        
        print("All data and visualization reset")
        self.check_ready_to_predict()
    
    def update_workflow_step(self, step_index, active, title=None):
        """Update a workflow step indicator."""
        if step_index >= len(self.workflow_steps):
            return
            
        step_frame = self.workflow_steps[step_index]
        
        # Get the step number frame and content frame
        number_frame = step_frame.winfo_children()[0]  # First child is number frame
        content_frame = step_frame.winfo_children()[1]  # Second child is content frame
        
        # Update colors based on active state
        if active:
            number_frame.configure(fg_color=AppConfig.COLORS['success'])
            # Update number label color
            number_label = number_frame.winfo_children()[0]
            number_label.configure(text_color="white")
            
            # Update title label color
            title_label = content_frame.winfo_children()[0]
            title_label.configure(text_color=AppConfig.COLORS['text_primary'])
            if title:
                title_label.configure(text=title)
                
        else:
            number_frame.configure(fg_color=AppConfig.COLORS['border'])
            # Update number label color
            number_label = number_frame.winfo_children()[0]
            number_label.configure(text_color=AppConfig.COLORS['text_secondary'])
            
            # Update title label color
            title_label = content_frame.winfo_children()[0]
            title_label.configure(text_color=AppConfig.COLORS['text_secondary'])
            if title:
                title_label.configure(text=title)
    
    def check_ready_to_predict(self):
        """Check if prediction requirements are met and update UI."""
        is_valid, message = self.data_validator.validate_prediction_requirements(
            self.flair_data, self.t1ce_data, self.model
        )
        
        print(f"Checking readiness: {message}")
        
        if is_valid:
            self.predict_btn.configure(state="normal")
            self.update_workflow_step(2, True, "🧠 Ready to Predict")
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
            self.update_workflow_step(3, True, "📊 Ready to Visualize")
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
        
        # Update UI for prediction start
        self.predict_btn.configure(
            state="disabled", 
            text="🔄 Analyzing...",
            fg_color=AppConfig.COLORS['warning']
        )
        self.status_label.configure(
            text="🔄 Running AI analysis - This may take a few minutes...",
            text_color=AppConfig.COLORS['warning']
        )
        self.info_label.configure(text="AI is analyzing your brain images. Please wait...")
        
        # Update workflow step
        self.update_workflow_step(2, True, "🔄 Analyzing Images...")
        
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
        self.predict_btn.configure(
            state="normal", 
            text="🧠 Predict Segmentation",
            fg_color=AppConfig.COLORS['success']
        )
        self.visualize_btn.configure(state="normal")
        
        self.status_label.configure(
            text="✅ Analysis completed successfully - Results ready for visualization",
            text_color=AppConfig.COLORS['success']
        )
        self.info_label.configure(text="✅ Analysis completed! Click 'Visualize' to explore the results.")
        
        # Update workflow steps
        self.update_workflow_step(2, True, "✅ Analysis Complete")
        self.update_workflow_step(3, True, "📊 Ready to Visualize")
        
        if self.predictions is not None:
            # Get prediction summary
            summary = self.model.get_prediction_summary(self.predictions, self.threshold)
            print(f"Opening visualization with predictions shape={summary['shape']}")
            
            # Check if predictions are above threshold
            above_threshold = any(summary['above_threshold'].values())
            if not above_threshold:
                self.info_label.configure(
                    text="⚠️ Predictions are below threshold. Try lowering the threshold in visualization."
                )
                messagebox.showwarning(
                    "Low Confidence Results", 
                    "The AI predictions are below the current threshold. You can explore the results and adjust the threshold in the visualization window."
                )
            
            # Auto-open visualization
            self.viz_window.create_window()
        else:
            print("No predictions available to visualize")
            self.status_label.configure(
                text="❌ Error: No predictions generated",
                text_color=AppConfig.COLORS['error']
            )
            self.info_label.configure(text="❌ Error: No predictions generated. Please try again.")
            messagebox.showerror("Error", "No predictions generated. Please try again.")
    
    def _prediction_error(self, error_msg):
        """Handle prediction error in main thread."""
        self.predict_btn.configure(
            state="normal", 
            text="🧠 Predict Segmentation",
            fg_color=AppConfig.COLORS['success']
        )
        self.status_label.configure(
            text=f"❌ Analysis failed: {error_msg}",
            text_color=AppConfig.COLORS['error']
        )
        self.info_label.configure(text=f"❌ Analysis failed: {error_msg}")
        
        # Reset workflow step
        self.update_workflow_step(2, True, "❌ Analysis Failed")
        
        messagebox.showerror("Analysis Error", f"Failed to run prediction: {error_msg}")
    
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
