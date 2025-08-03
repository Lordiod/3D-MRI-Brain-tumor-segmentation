@echo off
REM GPU Environment Setup Script for Brain Tumor Segmentation Training (Windows)
REM Usage: setup_gpu_env.bat

echo 🚀 Setting up GPU environment for Brain Tumor Segmentation Training...

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% == 0 (
    echo ✅ Conda found. Creating conda environment...
    
    REM Create conda environment
    conda create -n brain-tumor-gpu python=3.9 -y
    
    REM Activate environment
    echo 📦 Activating environment...
    call conda activate brain-tumor-gpu
    
    REM Install CUDA and cuDNN
    echo 🔧 Installing CUDA and cuDNN...
    conda install cudatoolkit=11.2 cudnn=8.1.0 -c conda-forge -y
    
    REM Install TensorFlow GPU
    echo 🤖 Installing TensorFlow with GPU support...
    pip install tensorflow==2.10.0
    
    REM Install requirements
    echo 📋 Installing requirements...
    pip install -r requirements-gpu.txt
    
    echo ✅ Conda environment 'brain-tumor-gpu' created successfully!
    echo 💡 Activate with: conda activate brain-tumor-gpu
    
) else (
    echo 🐍 Conda not found. Creating virtual environment with pip...
    
    REM Create virtual environment
    python -m venv tf-gpu-env
    
    REM Activate environment
    call tf-gpu-env\Scripts\activate.bat
    
    REM Upgrade pip
    python -m pip install --upgrade pip
    
    REM Install TensorFlow GPU
    echo 🤖 Installing TensorFlow with GPU support...
    pip install tensorflow==2.10.0
    
    REM Install requirements
    echo 📋 Installing requirements...
    pip install -r requirements-gpu.txt
    
    echo ✅ Virtual environment 'tf-gpu-env' created successfully!
    echo 💡 Activate with: tf-gpu-env\Scripts\activate.bat
)

REM Verify GPU setup
echo 🔍 Verifying GPU setup...
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print('GPU devices available:', len(gpus)); [print(f'  - {gpu}') for gpu in gpus]; print('✅ GPU setup successful!' if len(gpus) > 0 else '⚠️  No GPU detected. Please check your CUDA installation.')"

echo.
echo 🎯 Next steps:
echo 1. Download the BraTS2020 dataset:
echo    kaggle datasets download -d awsaf49/brats20-dataset-training-validation
echo 2. Extract to Data\ directory
echo 3. Open the training notebook:
echo    jupyter notebook "building the model\3d-mri-brain-tumor-segmentation-u-net.ipynb"
echo.
echo 💡 For dataset download, install Kaggle CLI:
echo    pip install kaggle
echo    Then configure your Kaggle API credentials
echo.
echo 🚀 Happy training!

pause
