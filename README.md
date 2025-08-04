# 3D MRI Brain Tumor Segmentation GUI Application



[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A modern GUI application built with CustomTkinter that uses a trained U-Net deep learning model to perform automated brain tumor segmentation on NIfTI MRI data. This tool provides medical professionals and researchers with an intuitive interface for analyzing brain tumor patterns in 3D MRI scans.

![3D MRI Brain Tumor Segmentation GUI](public/app.gif)

## ✨ Features

- **Multi-sequence MRI Support**: Load FLAIR and T1CE NIfTI images
- **Real-time AI Predictions**: Instant tumor segmentation using trained U-Net model  
- **Interactive 3D Navigation**: Browse through all slices of 3D volumes with smooth slider controls
- **Multi-class Visualization**: Visual display of different tumor classes:
  - 🔴 Necrotic/Core regions
  - 🟡 Edema areas  
  - 🟢 Enhancing tumor tissue
  - ⚫ Background/healthy tissue
- **Modern Interface**: Clean, dark-themed GUI built with CustomTkinter
- **Modular Architecture**: Well-organized codebase for easy maintenance and extension

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Windows, macOS, or Linux
- At least 4GB RAM (8GB recommended for larger datasets)
- **For model training**: NVIDIA GPU with CUDA support (highly recommended)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/3d-mri-brain-tumor-segmentation.git
   cd 3d-mri-brain-tumor-segmentation
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download or train a model**:
   - **Option A**: Train your own model using the Jupyter notebook in `building the model/`
     - Download the BraTS2020 dataset from [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
     - Set up a GPU environment (see [GPU Setup Guide](#-gpu-setup-for-training) below)
   - **Option B**: Place a compatible pre-trained model as `model_x1_1.h5` in the root directory
   - See [model_placeholder.txt](model_placeholder.txt) for details

### Running the Application

```bash
python main.py
```

## 📖 Usage Guide

1. **Launch the application**:
   ```bash
   python main.py
   ```

2. **Load MRI Images**:
   - Click **"Load FLAIR Image"** to select a FLAIR NIfTI file (.nii or .nii.gz)
   - Click **"Load T1CE Image"** to select a T1CE NIfTI file (.nii or .nii.gz)
   - Both sequences are required for accurate segmentation

3. **Generate Predictions**:
   - Once both images are loaded, click **"Predict Segmentation"**
   - The AI model will process the images and display tumor segmentation results
   - Processing time depends on your hardware (typically 10-30 seconds)

4. **Explore Results**:
   - Use the **slice slider** to navigate through different slices of the 3D volume
   - View the **6-panel display**:
     - Top row: Original FLAIR, Original T1CE, Combined prediction overlay
     - Bottom row: Individual tumor class predictions (Necrotic, Edema, Enhancing)

## 🧠 Model Information

This application uses a **3D U-Net architecture** specifically designed for multi-class brain tumor segmentation:

- **Architecture**: Deep convolutional neural network with encoder-decoder structure
- **Training Data**: BraTS (Brain Tumor Segmentation) challenge dataset
- **Input**: Dual-channel MRI (FLAIR + T1CE sequences)
- **Output**: 4-class probability maps

### Segmentation Classes

| Class | Label | Description | Color |
|-------|-------|-------------|-------|
| 0 | Background | Healthy brain tissue | Black |
| 1 | Necrotic/Core | Non-viable tumor core | Red |
| 2 | Edema | Peritumoral swelling | Yellow |
| 3 | Enhancing | Active tumor tissue | Green |

## 📋 Input Requirements

- **File Format**: NIfTI (.nii or .nii.gz)
- **Required Sequences**: Both FLAIR and T1CE MRI sequences
- **Preprocessing**: Images should ideally be:
  - Skull-stripped (non-brain tissue removed)
  - Co-registered (aligned between sequences)  
  - Bias field corrected
- **Dimensions**: Any size (automatically resized to 64x64 for model input)

## 🖥️ System Requirements

### For Running the Application (Inference)

**Minimum Requirements**
- **OS**: Windows 10, macOS 10.14, or Linux (Ubuntu 18.04+)
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Python**: 3.8 or higher

**Recommended Requirements**
- **RAM**: 8GB or more
- **GPU**: NVIDIA GPU with CUDA support (for faster inference)
- **Storage**: 5GB free space (for datasets)

### For Training the Model

**Essential Requirements**
- **GPU**: NVIDIA GPU with at least 8GB VRAM (RTX 3070/4060 Ti or better)
- **RAM**: 16GB+ system RAM (32GB recommended)
- **Storage**: 50GB+ free space (dataset + checkpoints + logs)
- **CUDA**: Version 11.2 or higher
- **Python**: 3.8-3.10 (3.9 recommended for best compatibility)

**Recommended Training Setup**
- **GPU**: RTX 4070/4080, RTX 3080/3090, or Tesla V100+
- **RAM**: 32GB or more
- **Storage**: SSD with 100GB+ free space
- **Internet**: High-speed connection for dataset download (~6GB)

## 🏗️ Project Structure

```
3d-mri-brain-tumor-segmentation/
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
├── model_placeholder.txt      # Model download instructions
├── models/                    # Model handling and prediction
│   ├── __init__.py
│   └── brain_tumor_model.py
├── ui/                        # User interface components  
│   ├── __init__.py
│   ├── main_window.py
│   └── visualization_window.py
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── config.py
│   └── data_handler.py
└── building the model/        # Training resources
    ├── 3d-mri-brain-tumor-segmentation-u-net.ipynb
    └── model.png
```

## 🎯 GPU Setup for Training

Training the U-Net model requires significant computational resources. A GPU setup is **essential** for reasonable training times.

### Requirements for Training
- **GPU**: NVIDIA GPU with at least 8GB VRAM (RTX 3070/4060 or better recommended)
- **CUDA**: Version 11.2 or higher
- **cuDNN**: Compatible version with CUDA
- **RAM**: At least 16GB system RAM
- **Storage**: 50GB+ free space for dataset and model checkpoints

### Setting Up GPU Environment

#### Option 1: Manual CUDA & cuDNN Installation (Windows)

**🔹 1. Uninstall Any Existing CUDA & cuDNN**

First, clean your environment to avoid conflicts:
- Open **Control Panel > Programs > Programs and Features**
- Uninstall **NVIDIA CUDA Toolkit**
- Uninstall any **cuDNN**, if manually installed before

**🔹 2. Install CUDA Toolkit 11.2**

Official NVIDIA download page: https://developer.nvidia.com/cuda-11.2.2-download-archive

Choose:
- **Operating System**: Windows
- **Architecture**: x86_64
- **Version**: Windows 10
- **Installer Type**: exe (local)

Download and install the toolkit.

🔧 **Make sure to add CUDA to your system's environment variables. The installer usually does this for you.**

**🔹 3. Install cuDNN 8.1 for CUDA 11.2**

⚠️ **You need an NVIDIA developer account to download cuDNN**

1. Go to: https://developer.nvidia.com/rdp/cudnn-archive
2. Look for: **cuDNN v8.1.1 (February 26th, 2021), for CUDA 11.2**
3. Download the cuDNN for Windows zip file

After download:
1. Extract the zip file
2. Copy the contents to your CUDA directory:
   - Copy `bin/*` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`
   - Copy `include/*` → `...\CUDA\v11.2\include`
   - Copy `lib/*` → `...\CUDA\v11.2\lib\x64`

🔒 **When asked to replace files, click Yes.**

**🔹 4. Add Environment Variables (if not already set)**

Open **System Properties > Environment Variables**, then add to System variables:

- **CUDA_HOME** or **CUDA_PATH** → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`

Append to **Path**:
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp`

**🔹 5. Install TensorFlow Compatible Version**

Now install a version of TensorFlow that works with CUDA 11.2 + cuDNN 8.1:

```bash
pip install tensorflow==2.7.0
```

#### Option 2: Using Conda (Alternative)

```bash
# Create a new conda environment
conda create -n brain-tumor-gpu python=3.9
conda activate brain-tumor-gpu

# Install CUDA and cuDNN through conda
conda install cudatoolkit=11.2 cudnn=8.1.0 -c conda-forge

# Install TensorFlow with GPU support
pip install tensorflow==2.10.0

# Install other requirements
pip install -r requirements.txt

# Verify GPU setup
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

#### Option 3: Using pip with system CUDA

```bash
# Create virtual environment
python -m venv tf-gpu-env
source tf-gpu-env/bin/activate  # On Windows: tf-gpu-env\Scripts\activate

# Install TensorFlow GPU
pip install tensorflow==2.10.0

# Install requirements
pip install -r requirements.txt

# Verify GPU setup
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

### Dataset Download and Setup

1. **Download BraTS2020 Dataset**:
   ```bash
   # Install Kaggle CLI
   pip install kaggle
   
   # Configure Kaggle credentials (follow Kaggle API setup)
   # Download dataset
   kaggle datasets download -d awsaf49/brats20-dataset-training-validation
   
   # Extract to Data/ directory
   unzip brats20-dataset-training-validation.zip -d Data/
   ```

2. **Dataset Structure**:
   ```
   Data/
   └── BraTS2020_TrainingData/
       └── MICCAI_BraTS2020_TrainingData/
           ├── BraTS20_Training_001/
           ├── BraTS20_Training_002/
           └── ... (369 training cases)
   ```

### Training the Model

1. **Open the training notebook**:
   ```bash
   # Activate GPU environment
   conda activate brain-tumor-gpu  # or source tf-gpu-env/bin/activate
   
   # Start Jupyter
   jupyter notebook "building the model/3d-mri-brain-tumor-segmentation-u-net.ipynb"
   ```

2. **Training Tips**:
   - **Batch Size**: Start with batch_size=2 for 8GB VRAM, increase if you have more memory
   - **Mixed Precision**: Enable for faster training: `tf.keras.mixed_precision.set_global_policy('mixed_float16')`
   - **Checkpoints**: Save model checkpoints frequently during training
   - **Expected Time**: 24-48 hours for full training on RTX 3080/4070

3. **Monitor GPU Usage**:
   ```bash
   # Monitor GPU utilization
   nvidia-smi -l 1  # Updates every second
   
   # Check GPU memory in Python
   python -c "import tensorflow as tf; print(tf.config.experimental.get_memory_info('GPU:0'))"
   ```

### Troubleshooting GPU Setup

**Common Issues**:

1. **CUDA Version Mismatch**:
   ```bash
   # Check CUDA version
   nvidia-smi  # Look at CUDA Version in top right
   
   # Install compatible TensorFlow version
   # CUDA 11.2 -> tensorflow==2.10.0
   # CUDA 11.8+ -> tensorflow>=2.12.0
   ```

2. **Out of Memory Errors**:
   - Reduce batch size in training notebook
   - Enable memory growth: `tf.config.experimental.set_memory_growth(gpu, True)`
   - Use gradient checkpointing to reduce memory usage

3. **GPU Not Detected**:
   ```bash
   # Reinstall TensorFlow GPU
   pip uninstall tensorflow
   pip install tensorflow==2.10.0
   
   # Verify installation
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
   ```

### Cloud Training Alternatives

If you don't have a suitable GPU, consider cloud options:

- **Google Colab Pro/Pro+**: GPU/TPU access with pre-configured environment
- **Kaggle Notebooks**: Free GPU time with direct dataset access
- **AWS EC2 P-series**: p3.2xlarge or p3.8xlarge instances
- **Google Cloud AI Platform**: Pre-configured ML environments

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:
- Setting up the development environment
- Code style guidelines  
- Submitting pull requests
- Reporting issues

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BraTS Challenge**: For providing the brain tumor segmentation dataset
- **TensorFlow/Keras**: For the deep learning framework
- **CustomTkinter**: For the modern GUI components
- **NiBabel**: For NIfTI file handling
- **Medical Imaging Community**: For advancing open-source medical AI tools

## 📞 Support

- **Issues**: Please use the [GitHub Issues](https://github.com/your-username/3d-mri-brain-tumor-segmentation/issues) page
- **Discussions**: Join our [GitHub Discussions](https://github.com/your-username/3d-mri-brain-tumor-segmentation/discussions)
- **Documentation**: See the [Wiki](https://github.com/your-username/3d-mri-brain-tumor-segmentation/wiki) for detailed guides

## ⚠️ Medical Disclaimer

This software is for research and educational purposes only. It is not intended for clinical diagnosis or treatment decisions. Always consult qualified medical professionals for medical advice and diagnosis.

---

**Star ⭐ this repository if you find it helpful!**
