#!/usr/bin/env python3
"""
Google Colab setup script for SEMA inference
Run this in a Colab cell to set up the environment
"""

def setup_colab_environment():
    """Setup Google Colab environment for SEMA inference"""
    
    print("🚀 Setting up SEMA inference for Google Colab...")
    
    # Install system dependencies
    print("📦 Installing system dependencies...")
    import subprocess
    import sys
    
    try:
        # Install Java for KoNLPy
        subprocess.run(['apt-get', 'update', '-qq'], check=True)
        subprocess.run(['apt-get', 'install', '-y', '-qq', 'openjdk-8-jdk'], check=True)
        
        # Set Java environment
        import os
        os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64'
        
        print("✅ Java installed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing Java: {e}")
        return False
    
    # Install Python packages
    print("📦 Installing Python packages...")
    packages = [
        'torch',
        'transformers>=4.30.0',
        'huggingface_hub',
        'konlpy',
        'pandas',
        'numpy',
        'scikit-learn',
        'openpyxl',
        'tqdm',
        'torchmetrics'
    ]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                         check=True, capture_output=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing {package}: {e}")
            return False
    
    print("🎉 Setup complete!")
    return True

def upload_required_files():
    """Upload required data files to Colab"""
    from google.colab import files
    import os
    
    print("📤 Please upload the following required files:")
    print("- data2.pkl (MultiLabelBinarizer)")
    print("- voc_etc.pkl (Filter data)")  
    print("- keyword_doc.pkl (Keywords)")
    print("- model checkpoint (.ckpt file)")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    print("Click 'Choose Files' to upload:")
    uploaded = files.upload()
    
    # Move uploaded files to data directory
    for filename in uploaded.keys():
        if filename.endswith('.pkl'):
            import shutil
            shutil.move(filename, f'data/{filename}')
            print(f"✅ Moved {filename} to data/")
    
    return list(uploaded.keys())

def verify_setup():
    """Verify that everything is set up correctly"""
    print("🔍 Verifying setup...")
    
    try:
        # Test Java
        import subprocess
        result = subprocess.run(['java', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Java is working")
        else:
            print("❌ Java not working")
            return False
            
        # Test KoNLPy
        from konlpy.tag import Kkma
        kkma = Kkma()
        test_result = kkma.morphs("테스트")
        print("✅ KoNLPy is working")
        
        # Test PyTorch
        import torch
        print(f"✅ PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}")
        
        # Test transformers
        from transformers import AutoTokenizer
        print("✅ Transformers is working")
        
        # Check data files
        import os
        required_files = ['data/data2.pkl', 'data/voc_etc.pkl', 'data/keyword_doc.pkl']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"⚠️ Missing files: {missing_files}")
            print("Please upload these files using upload_required_files()")
            return False
        else:
            print("✅ All required data files found")
        
        print("🎉 All checks passed! Ready to run SEMA inference.")
        return True
        
    except Exception as e:
        print(f"❌ Setup verification failed: {e}")
        return False

def run_colab_setup():
    """Complete setup process for Google Colab"""
    print("🔧 Starting Google Colab setup for SEMA inference...")
    
    # Step 1: Setup environment
    if not setup_colab_environment():
        print("❌ Environment setup failed")
        return False
    
    # Step 2: Upload files
    print("\n📁 Step 2: Upload required files")
    print("You need to upload the model files manually.")
    print("Run upload_required_files() in the next cell after this completes.")
    
    return True

if __name__ == "__main__":
    # Instructions for Colab usage
    print("""
    🚀 SEMA Inference - Google Colab Setup
    
    Instructions:
    1. Run this cell to set up the environment
    2. Run upload_required_files() to upload your model files
    3. Run verify_setup() to check everything is working
    4. Then you can use the SEMA inference CLI
    
    Example usage after setup:
    
    from src.cli import SemaInference
    
    # Initialize (will use uploaded files in data/ folder)
    inferencer = SemaInference()
    
    # Process a file
    inferencer.process_file('input.xlsx', 'output.xlsx')
    """)