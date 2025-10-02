"""
Setup script for VoiceScore dependencies
Run this before using the main pipeline
"""

import subprocess
import sys
import logging

def install_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Python requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    return True

def download_spacy_model():
    """Download spaCy English model"""
    print("Downloading spaCy English model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✓ spaCy model downloaded successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download spaCy model: {e}")
        return False
    return True

def setup_nltk_data():
    """Setup NLTK data"""
    print("Setting up NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✓ NLTK data downloaded successfully")
    except Exception as e:
        print(f"❌ Failed to setup NLTK data: {e}")
        return False
    return True

def verify_pytorch():
    """Verify PyTorch installation"""
    print("Verifying PyTorch installation...")
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError:
        print("❌ PyTorch not found")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up VoiceScore...")
    print("=" * 50)

    success = True

    # Install requirements
    if not install_requirements():
        success = False

    # Download spaCy model
    if not download_spacy_model():
        success = False

    # Setup NLTK
    if not setup_nltk_data():
        success = False

    # Verify PyTorch
    if not verify_pytorch():
        success = False

    print("=" * 50)
    if success:
        print("🎉 VoiceScore setup completed successfully!")
        print("\nYou can now run:")
        print("  python voicescore_pipeline.py")
    else:
        print("❌ Setup completed with errors. Please check the messages above.")

if __name__ == "__main__":
    main()


