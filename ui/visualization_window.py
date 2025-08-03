"""
Visualization window for displaying segmentation results.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
import tkinter as tk

from utils.config import AppConfig


class VisualizationWindow:
    """Handle the visualization window for displaying segmentation results."""
    
    def __init__(self, app):
        self.app = app
        self.window = None
        self.fig = None
        self.axes = None
        self.canvas = None
        
        # Control variables
        self.slice_var = None
        self.threshold_var = None
        self.slice_scale = None
        self.threshold_scale = None
    
    def create_window(self):
        """Create the visualization window."""
        if self.window is not None:
            self.window.destroy()
        
        self.window = tk.Toplevel(self.app.root)
        self.window.title("Brain Tumor Segmentation Results")
        self.window.geometry(f"{AppConfig.VISUALIZATION_WIDTH}x{AppConfig.VISUALIZATION_HEIGHT}")
        self.window.configure(bg='#2b2b2b')
        
        # Ensure window is brought to front
        self.window.attributes('-topmost', True)
        self.window.update()
        self.window.attributes('-topmost', False)
        
        # Create matplotlib figure
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(2, 3, figsize=(16, 10))
        self.fig.patch.set_facecolor('#2b2b2b')
        
        # Configure axes
        for ax in self.axes.flat:
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
        
        # Create control frame
        self._create_controls()
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)
        
        # Initial display
        self.update_display()
        print("Visualization window created and updated")
    
    def _create_controls(self):
        """Create control widgets for slice and threshold adjustment."""
        control_frame = tk.Frame(self.window, bg='#2b2b2b')
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Slice control
        tk.Label(control_frame, text="Slice:", bg='#2b2b2b', fg='white').pack(side='left', padx=5)
        
        self.slice_var = tk.IntVar(value=self.app.current_slice)
        self.slice_scale = tk.Scale(
            control_frame,
            from_=0,
            to=AppConfig.VOLUME_SLICES-1,
            orient='horizontal',
            variable=self.slice_var,
            command=self.update_slice,
            bg='#2b2b2b',
            fg='white',
            highlightbackground='#2b2b2b'
        )
        self.slice_scale.pack(side='left', fill='x', expand=True, padx=5)
        
        # Threshold control
        tk.Label(control_frame, text="Threshold:", bg='#2b2b2b', fg='white').pack(side='left', padx=5)
        
        self.threshold_var = tk.DoubleVar(value=self.app.threshold)
        self.threshold_scale = tk.Scale(
            control_frame,
            from_=0.1,
            to=0.9,
            resolution=0.01,
            orient='horizontal',
            variable=self.threshold_var,
            command=self.update_threshold,
            bg='#2b2b2b',
            fg='white',
            highlightbackground='#2b2b2b'
        )
        self.threshold_scale.pack(side='left', fill='x', expand=True, padx=5)
    
    def update_slice(self, value):
        """Update the current slice."""
        self.app.current_slice = int(value)
        self.update_display()
    
    def update_threshold(self, value):
        """Update the prediction threshold."""
        self.app.threshold = float(value)
        self.update_display()
    
    def update_display(self):
        """Update the visualization display."""
        if self.fig is None or self.axes is None or self.window is None:
            print("VisualizationWindow: Cannot update display - figure, axes, or window not initialized")
            return
        
        # Clear all subplots
        for ax in self.axes.flat:
            ax.clear()
            ax.set_facecolor('#2b2b2b')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
        
        slice_idx = self.app.current_slice + AppConfig.VOLUME_START_AT
        
        try:
            # Display FLAIR image
            self._display_flair_image(slice_idx)
            
            # Display T1CE image
            self._display_t1ce_image(slice_idx)
            
            # Display combined overlay
            self._display_combined_prediction(slice_idx)
            
            # Display individual segmentation classes
            self._display_individual_classes(slice_idx)
            
        except Exception as e:
            print(f"Visualization error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Update canvas
        try:
            self.fig.tight_layout()
            self.canvas.draw()
            self.canvas.flush_events()
            self.window.update()
        except Exception as e:
            print(f"Canvas update error: {str(e)}")
    
    def _display_flair_image(self, slice_idx):
        """Display FLAIR image in the visualization."""
        if self.app.flair_data is not None and slice_idx < self.app.flair_data.shape[2]:
            flair_slice = self.app.flair_data[:, :, slice_idx]
            flair_normalized = self._normalize_image(flair_slice)
            
            print(f"Displaying FLAIR slice {slice_idx}, shape={flair_slice.shape}")
            self.axes[0, 0].imshow(flair_normalized, cmap='gray', aspect='auto')
            self.axes[0, 0].set_title('FLAIR', color='white', fontsize=12)
            self.axes[0, 0].axis('off')
        else:
            print(f"No FLAIR data or invalid slice index: {slice_idx}")
            self._display_no_data_message(self.axes[0, 0], 'No FLAIR Data', 'FLAIR')
    
    def _display_t1ce_image(self, slice_idx):
        """Display T1CE image in the visualization."""
        if self.app.t1ce_data is not None and slice_idx < self.app.t1ce_data.shape[2]:
            t1ce_slice = self.app.t1ce_data[:, :, slice_idx]
            t1ce_normalized = self._normalize_image(t1ce_slice)
            
            print(f"Displaying T1CE slice {slice_idx}, shape={t1ce_slice.shape}")
            self.axes[0, 1].imshow(t1ce_normalized, cmap='gray', aspect='auto')
            self.axes[0, 1].set_title('T1CE', color='white', fontsize=12)
            self.axes[0, 1].axis('off')
        else:
            print(f"No T1CE data or invalid slice index: {slice_idx}")
            self._display_no_data_message(self.axes[0, 1], 'No T1CE Data', 'T1CE')
    
    def _display_combined_prediction(self, slice_idx):
        """Display combined prediction overlay."""
        if self.app.flair_data is not None and slice_idx < self.app.flair_data.shape[2]:
            flair_slice = self.app.flair_data[:, :, slice_idx]
            flair_resized = cv2.resize(flair_slice, (AppConfig.IMG_SIZE, AppConfig.IMG_SIZE))
            flair_normalized = self._normalize_image(flair_resized)
            
            print(f"Displaying combined prediction on FLAIR background, slice {slice_idx}")
            self.axes[0, 2].imshow(flair_normalized, cmap='gray', aspect='auto')
            
            if self.app.predictions is not None and self.app.current_slice < self.app.predictions.shape[0]:
                self._overlay_combined_predictions()
            
            self.axes[0, 2].set_title('Combined Prediction', color='white', fontsize=12)
            self.axes[0, 2].axis('off')
        else:
            print("No FLAIR data for combined prediction")
            self._display_no_data_message(self.axes[0, 2], 'No Prediction Data', 'Combined Prediction')
    
    def _display_individual_classes(self, slice_idx):
        """Display individual segmentation classes."""
        if (self.app.predictions is not None and 
            self.app.current_slice < self.app.predictions.shape[0] and
            self.app.flair_data is not None and 
            slice_idx < self.app.flair_data.shape[2]):
            
            flair_slice = self.app.flair_data[:, :, slice_idx]
            flair_resized = cv2.resize(flair_slice, (AppConfig.IMG_SIZE, AppConfig.IMG_SIZE))
            flair_normalized = self._normalize_image(flair_resized)
            
            class_colormaps = AppConfig.get_class_colormaps()
            
            for i in range(3):
                class_idx = i + 1
                class_name = AppConfig.SEGMENT_CLASSES[class_idx]
                
                self.axes[1, i].imshow(flair_normalized, cmap='gray', aspect='auto')
                
                pred_class = self.app.predictions[self.app.current_slice, :, :, class_idx]
                print(f"Class {class_name} prediction range: {np.min(pred_class)} to {np.max(pred_class)}")
                
                thresholded_pred = np.where(pred_class > self.app.threshold, pred_class, 0)
                
                if np.any(thresholded_pred > 0):
                    masked_pred = np.ma.masked_where(thresholded_pred == 0, thresholded_pred)
                    self.axes[1, i].imshow(
                        masked_pred, 
                        cmap=class_colormaps[class_idx], 
                        alpha=0.7, 
                        vmin=0, 
                        vmax=1, 
                        aspect='auto'
                    )
                    print(f"Displayed {class_name} mask with non-zero values: {np.sum(thresholded_pred > 0)}")
                else:
                    print(f"No significant {class_name} predictions above threshold")
                    self._display_no_data_message(
                        self.axes[1, i], 
                        f'No {class_name} Prediction', 
                        class_name,
                        fontsize=10
                    )
                
                self.axes[1, i].set_title(class_name, color='white', fontsize=12)
                self.axes[1, i].axis('off')
        else:
            print("No prediction data for individual classes")
            class_names = [AppConfig.SEGMENT_CLASSES[i+1] for i in range(3)]
            for i, class_name in enumerate(class_names):
                self._display_no_data_message(
                    self.axes[1, i], 
                    'No Prediction', 
                    class_name
                )
    
    def _overlay_combined_predictions(self):
        """Overlay combined predictions on the current display."""
        pred_slice = self.app.predictions[self.app.current_slice, :, :, :]
        print(f"Prediction slice shape: {pred_slice.shape}, range: {np.min(pred_slice)} to {np.max(pred_slice)}")
        
        threshold = self.app.threshold
        combined_mask = np.zeros((AppConfig.IMG_SIZE, AppConfig.IMG_SIZE))
        
        # Apply threshold for each class (order matters for overlay)
        necrotic_mask = pred_slice[:, :, 1] > threshold
        combined_mask[necrotic_mask] = 1
        
        edema_mask = pred_slice[:, :, 2] > threshold
        combined_mask[edema_mask] = 2
        
        enhancing_mask = pred_slice[:, :, 3] > threshold
        combined_mask[enhancing_mask] = 3
        
        if np.any(combined_mask > 0):
            colors = list(AppConfig.get_class_colors().values())
            cmap = ListedColormap(colors)
            masked_predictions = np.ma.masked_where(combined_mask == 0, combined_mask)
            self.axes[0, 2].imshow(
                masked_predictions, 
                cmap=cmap, 
                alpha=0.6, 
                vmin=0, 
                vmax=3, 
                aspect='auto'
            )
            print(f"Displayed combined mask with non-zero values: {np.sum(combined_mask > 0)}")
        else:
            print("No significant predictions above threshold")
            self.axes[0, 2].text(
                0.5, 0.5, 
                'No Significant Predictions',
                ha='center', va='center', 
                transform=self.axes[0, 2].transAxes,
                color='white', fontsize=10
            )
    
    def _normalize_image(self, image):
        """Normalize image data to 0-1 range."""
        if np.max(image) > np.min(image):
            return (image - np.min(image)) / (np.max(image) - np.min(image))
        return image
    
    def _display_no_data_message(self, ax, message, title, fontsize=12):
        """Display a no data message on the given axis."""
        ax.text(
            0.5, 0.5, 
            message,
            ha='center', va='center', 
            transform=ax.transAxes,
            color='white', fontsize=fontsize
        )
        ax.set_title(title, color='white', fontsize=12)
