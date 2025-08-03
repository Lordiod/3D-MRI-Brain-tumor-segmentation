#!/bin/bash
# GPU Environment Setup Script for Brain Tumor Segmentation Training
# Usage: bash setup_gpu_env.sh

set -e  # Exit on any error

echo "🚀 Setting up GPU environment for Brain Tumor Segmentation Training..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✅ Conda found. Creating conda environment..."
    
    # Create conda environment
    conda create -n brain-tumor-gpu python=3.9 -y
    
    # Activate environment
    echo "📦 Activating environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate brain-tumor-gpu
    
    # Install CUDA and cuDNN
    echo "🔧 Installing CUDA and cuDNN..."
    conda install cudatoolkit=11.2 cudnn=8.1.0 -c conda-forge -y
    
    # Install TensorFlow GPU
    echo "🤖 Installing TensorFlow with GPU support..."
    pip install tensorflow==2.10.0
    
    # Install requirements
    echo "📋 Installing requirements..."
    pip install -r requirements-gpu.txt
    
    echo "✅ Conda environment 'brain-tumor-gpu' created successfully!"
    echo "💡 Activate with: conda activate brain-tumor-gpu"
    
elif command -v python3 &> /dev/null; then
    echo "🐍 Conda not found. Creating virtual environment with pip..."
    
    # Create virtual environment
    python3 -m venv tf-gpu-env
    
    # Activate environment
    source tf-gpu-env/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install TensorFlow GPU
    echo "🤖 Installing TensorFlow with GPU support..."
    pip install tensorflow==2.10.0
    
    # Install requirements
    echo "📋 Installing requirements..."
    pip install -r requirements-gpu.txt
    
    echo "✅ Virtual environment 'tf-gpu-env' created successfully!"
    echo "💡 Activate with: source tf-gpu-env/bin/activate"
    
else
    echo "❌ Neither conda nor python3 found. Please install Python first."
    exit 1
fi

# Verify GPU setup
echo "🔍 Verifying GPU setup..."
python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPU devices available:', len(gpus))
for gpu in gpus:
    print(f'  - {gpu}')
if len(gpus) == 0:
    print('⚠️  No GPU detected. Please check your CUDA installation.')
else:
    print('✅ GPU setup successful!')
    # Test GPU memory
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
        print('✅ GPU computation test passed!')
    except Exception as e:
        print(f'⚠️  GPU computation test failed: {e}')
"

echo ""
echo "🎯 Next steps:"
echo "1. Download the BraTS2020 dataset:"
echo "   kaggle datasets download -d awsaf49/brats20-dataset-training-validation"
echo "2. Extract to Data/ directory"
echo "3. Open the training notebook:"
echo "   jupyter notebook 'building the model/3d-mri-brain-tumor-segmentation-u-net.ipynb'"
echo ""
echo "💡 For dataset download, install Kaggle CLI:"
echo "   pip install kaggle"
echo "   Then configure your Kaggle API credentials"
echo ""
echo "🚀 Happy training!"
