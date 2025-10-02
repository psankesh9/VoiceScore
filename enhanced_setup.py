"""
Enhanced Setup Script for VoiceScore with Neural Embeddings
Installs all dependencies for ECAPA-TDNN, CLAP, and neural coreference resolution
"""

import subprocess
import sys
import logging
import os
from pathlib import Path

def setup_logging():
    """Setup logging for the setup process"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("enhanced_setup.log")
        ]
    )

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_cuda_availability():
    """Check CUDA availability"""
    print("🔍 Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️  CUDA not available, will use CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not yet installed, will check CUDA after installation")
        return False

def install_pytorch():
    """Install PyTorch with appropriate CUDA support"""
    print("📦 Installing PyTorch...")
    try:
        # Try to detect CUDA version
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0 and 'CUDA Version' in result.stdout:
                print("✓ NVIDIA GPU detected, installing PyTorch with CUDA support")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--upgrade",
                    "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"
                ])
            else:
                raise subprocess.CalledProcessError(1, "nvidia-smi")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  No CUDA detected, installing CPU-only PyTorch")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade",
                "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"
            ])

        print("✓ PyTorch installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch: {e}")
        return False

def install_core_requirements():
    """Install core requirements"""
    print("📦 Installing core requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade",
            "transformers>=4.25.0",
            "TTS>=0.20.0",
            "requests",
            "beautifulsoup4",
            "tqdm",
            "nltk",
            "numpy",
            "soundfile",
            "librosa",
            "sentence-transformers",
            "faiss-cpu"
        ])
        print("✓ Core requirements installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install core requirements: {e}")
        return False

def install_enhanced_requirements():
    """Install enhanced AI models and dependencies"""
    print("📦 Installing enhanced AI models...")

    # SpeechBrain for ECAPA-TDNN
    try:
        print("  📥 Installing SpeechBrain (ECAPA-TDNN speaker embeddings)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "speechbrain"])
        print("  ✓ SpeechBrain installed")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ SpeechBrain installation failed: {e}")
        print("  ⚠️  Speaker embeddings will not be available")

    # LAION-CLAP for text-to-audio embeddings
    try:
        print("  📥 Installing LAION-CLAP (text-to-audio embeddings)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "laion-clap"])
        print("  ✓ LAION-CLAP installed")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ LAION-CLAP installation failed: {e}")
        print("  ⚠️  Text-to-audio matching will not be available")

    # spaCy for NLP
    try:
        print("  📥 Installing spaCy...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy>=3.4.0"])
        print("  ✓ spaCy installed")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ spaCy installation failed: {e}")
        return False

    # Neural coreference resolution
    try:
        print("  📥 Installing neural coreference models...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "neuralcoref"])
        print("  ✓ neuralcoref installed")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ neuralcoref installation failed: {e}")
        print("  ⚠️  Neural coreference resolution will not be available")

    # AllenNLP (optional advanced coreference)
    try:
        print("  📥 Installing AllenNLP (advanced coreference)...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "allennlp", "allennlp-models"
        ])
        print("  ✓ AllenNLP installed")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ AllenNLP installation failed: {e}")
        print("  ⚠️  Advanced coreference models will not be available")

    return True

def download_models():
    """Download required language models"""
    print("📥 Downloading language models...")

    # spaCy English model
    try:
        print("  📥 Downloading spaCy English model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("  ✓ spaCy English model downloaded")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed to download spaCy model: {e}")
        return False

    # NLTK data
    try:
        print("  📥 Setting up NLTK data...")
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("  ✓ NLTK data downloaded")
    except Exception as e:
        print(f"  ❌ Failed to setup NLTK: {e}")
        return False

    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = [
        "enhanced_voicescore_output",
        "voice_samples",
        "pretrained_models",
        "cache"
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✓ Created {directory}/")

    return True

def verify_installation():
    """Verify that all components are properly installed"""
    print("🔍 Verifying enhanced installation...")

    # Test core imports
    try:
        import torch
        import transformers
        import TTS
        import nltk
        import spacy
        print("✓ Core packages verified")
    except ImportError as e:
        print(f"❌ Core package import failed: {e}")
        return False

    # Test enhanced imports
    enhanced_modules = {
        'speechbrain': 'ECAPA-TDNN speaker embeddings',
        'laion_clap': 'CLAP text-to-audio embeddings',
        'neuralcoref': 'Neural coreference resolution',
        'allennlp': 'Advanced NLP models'
    }

    available_modules = []
    for module, description in enhanced_modules.items():
        try:
            __import__(module)
            print(f"✓ {description} available")
            available_modules.append(module)
        except ImportError:
            print(f"⚠️  {description} not available")

    # Test spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✓ spaCy English model loaded")
    except OSError:
        print("❌ spaCy English model not found")
        return False

    # Test PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA {torch.version.cuda} available")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available, using CPU")
    except Exception as e:
        print(f"❌ PyTorch verification failed: {e}")
        return False

    print(f"📊 Enhanced modules available: {len(available_modules)}/{len(enhanced_modules)}")
    return True

def create_test_script():
    """Create a test script to verify the installation"""
    print("📝 Creating test script...")

    test_script = """#!/usr/bin/env python3
'''
Enhanced VoiceScore Installation Test
Run this script to verify your installation is working correctly
'''

import sys
import logging

logging.basicConfig(level=logging.INFO)

def test_imports():
    '''Test all enhanced imports'''
    print("Testing enhanced imports...")

    # Core imports
    try:
        import torch
        import transformers
        import TTS
        import nltk
        import spacy
        print("✓ Core packages imported successfully")
    except ImportError as e:
        print(f"❌ Core import failed: {e}")
        return False

    # Enhanced imports
    enhanced_available = {}

    try:
        import speechbrain
        enhanced_available['speechbrain'] = True
        print("✓ SpeechBrain (ECAPA-TDNN) available")
    except ImportError:
        enhanced_available['speechbrain'] = False
        print("⚠️  SpeechBrain not available")

    try:
        import laion_clap
        enhanced_available['laion_clap'] = True
        print("✓ LAION-CLAP available")
    except ImportError:
        enhanced_available['laion_clap'] = False
        print("⚠️  LAION-CLAP not available")

    try:
        import neuralcoref
        enhanced_available['neuralcoref'] = True
        print("✓ neuralcoref available")
    except ImportError:
        enhanced_available['neuralcoref'] = False
        print("⚠️  neuralcoref not available")

    return enhanced_available

def test_models():
    '''Test model loading'''
    print("\\nTesting model loading...")

    # Test spaCy
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✓ spaCy English model loaded")
    except Exception as e:
        print(f"❌ spaCy model failed: {e}")
        return False

    # Test TTS
    try:
        from TTS.api import TTS
        tts = TTS("tts_models/en/ljspeech/tacotron2-DCA")
        print("✓ TTS model loaded")
    except Exception as e:
        print(f"❌ TTS model failed: {e}")
        return False

    return True

def test_enhanced_pipeline():
    '''Test enhanced pipeline components'''
    print("\\nTesting enhanced pipeline components...")

    try:
        from enhanced_voice_embeddings import EnhancedVoiceEmbeddingManager
        manager = EnhancedVoiceEmbeddingManager()
        print("✓ Enhanced voice embedding manager initialized")
    except Exception as e:
        print(f"❌ Voice embedding manager failed: {e}")
        return False

    try:
        from enhanced_coreference import EnhancedCoreferenceResolver
        resolver = EnhancedCoreferenceResolver(use_neural_coref=False)  # Test without neural models first
        print("✓ Enhanced coreference resolver initialized")
    except Exception as e:
        print(f"❌ Coreference resolver failed: {e}")
        return False

    try:
        from enhanced_tts_system import EnhancedTTSEngine
        engine = EnhancedTTSEngine(device="cpu")
        print("✓ Enhanced TTS engine initialized")
    except Exception as e:
        print(f"❌ TTS engine failed: {e}")
        return False

    return True

def main():
    print("🧪 Enhanced VoiceScore Installation Test")
    print("=" * 50)

    # Test imports
    enhanced_modules = test_imports()

    # Test models
    if not test_models():
        print("❌ Model testing failed")
        return

    # Test pipeline
    if not test_enhanced_pipeline():
        print("❌ Pipeline testing failed")
        return

    print("\\n" + "=" * 50)
    print("🎉 Enhanced VoiceScore installation test completed!")

    # Summary
    available_count = sum(1 for available in enhanced_modules.values() if available)
    total_count = len(enhanced_modules)

    print(f"📊 Enhanced features: {available_count}/{total_count} available")

    if available_count == total_count:
        print("🚀 All enhanced features available! You're ready to go.")
    elif available_count >= 2:
        print("✅ Sufficient enhanced features available for good performance.")
    else:
        print("⚠️  Limited enhanced features. Consider reinstalling missing components.")

    print("\\nTo get started, run:")
    print("  python enhanced_voicescore_pipeline.py")

if __name__ == "__main__":
    main()
"""

    with open("test_enhanced_installation.py", "w") as f:
        f.write(test_script)

    print("✓ Test script created: test_enhanced_installation.py")
    return True

def main():
    """Main enhanced setup function"""
    setup_logging()

    print("🚀 Enhanced VoiceScore Setup")
    print("=" * 60)
    print("Installing neural embeddings, advanced coreference, and enhanced TTS")
    print("=" * 60)

    success = True

    # Check prerequisites
    if not check_python_version():
        return False

    # Install PyTorch first (foundation for everything)
    if not install_pytorch():
        success = False

    # Install core requirements
    if not install_core_requirements():
        success = False

    # Install enhanced AI components
    if not install_enhanced_requirements():
        print("⚠️  Some enhanced features may not be available")

    # Download models
    if not download_models():
        success = False

    # Create directories
    if not create_directories():
        success = False

    # Verify installation
    print("=" * 60)
    if success:
        if verify_installation():
            print("🎉 Enhanced VoiceScore setup completed successfully!")

            # Create test script
            create_test_script()

            print("\n🧪 To test your installation:")
            print("  python test_enhanced_installation.py")
            print("\n🚀 To get started:")
            print("  python enhanced_voicescore_pipeline.py")
            print("\n📚 Read the enhanced documentation:")
            print("  ENHANCED_README.md")
        else:
            print("⚠️  Setup completed with some issues. Check the verification results above.")
    else:
        print("❌ Setup completed with errors. Please check the messages above.")
        print("\n🔧 For troubleshooting:")
        print("  - Check enhanced_setup.log for detailed error messages")
        print("  - Ensure you have a stable internet connection")
        print("  - Consider using a virtual environment")

if __name__ == "__main__":
    main()