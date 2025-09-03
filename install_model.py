#!/usr/bin/env python3
"""
SEMA Model Installation CLI
Automatically selects and installs the appropriate model based on GPU memory
"""

import argparse
import os
import sys
import subprocess
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path

class ModelInstaller:
    def __init__(self):
        self.models = {
            "small": {
                "repo": "shc443/sema_small",
                "min_gpu_mb": 0,
                "max_gpu_mb": 8192,  # 8GB
                "description": "Lightweight model for smaller GPUs (<8GB)"
            },
            "large": {
                "repo": "shc443/sema2025",
                "min_gpu_mb": 8192,  # 8GB+
                "max_gpu_mb": float('inf'),
                "description": "Full-size model for larger GPUs (8GB+)"
            }
        }
        
    def get_gpu_memory(self):
        """Get available GPU memory in MB"""
        if not torch.cuda.is_available():
            return 0
        
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            return gpu_memory / (1024 * 1024)  # Convert to MB
        except Exception as e:
            print(f"⚠️ Error detecting GPU: {e}")
            return 0
    
    def select_model(self, force_model=None):
        """Select appropriate model based on GPU memory"""
        if force_model:
            if force_model in self.models:
                return force_model, self.models[force_model]
            else:
                print(f"❌ Invalid model choice: {force_model}")
                print(f"Available models: {', '.join(self.models.keys())}")
                sys.exit(1)
        
        gpu_memory = self.get_gpu_memory()
        
        if gpu_memory == 0:
            print("⚠️ No GPU detected, using CPU-compatible small model")
            return "small", self.models["small"]
        
        print(f"🔍 Detected GPU memory: {gpu_memory:.0f}MB")
        
        for model_name, config in self.models.items():
            if config["min_gpu_mb"] <= gpu_memory < config["max_gpu_mb"]:
                print(f"✅ Selected {model_name} model: {config['description']}")
                return model_name, config
        
        # Default to large if GPU is very large
        return "large", self.models["large"]
    
    def download_model(self, repo_id, local_dir="./model"):
        """Download model from Hugging Face"""
        try:
            print(f"📥 Downloading model from {repo_id}...")
            
            # Create local directory
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            
            # Download the entire repository
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                resume_download=True
            )
            
            print(f"✅ Model downloaded to {local_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            return False
    
    def install_dependencies(self):
        """Install required dependencies"""
        print("📦 Installing dependencies...")
        
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True, capture_output=True, text=True)
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing dependencies: {e.stderr}")
            return False
    
    def setup_data_files(self):
        """Ensure required data files are present"""
        required_files = [
            "data/data2.pkl",
            "data/voc_etc.pkl", 
            "data/keyword_doc.pkl"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print("⚠️ Missing required data files:")
            for file in missing_files:
                print(f"  - {file}")
            print("These files should be downloaded manually from your data source")
            return False
        
        print("✅ All required data files present")
        return True

def main():
    parser = argparse.ArgumentParser(description='SEMA Model Installation CLI')
    parser.add_argument('-m', '--model', choices=['small', 'large'], 
                       help='Force specific model selection (overrides GPU detection)')
    parser.add_argument('-d', '--dir', default='./model',
                       help='Model download directory (default: ./model)')
    parser.add_argument('--no-deps', action='store_true',
                       help='Skip dependency installation')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check GPU and show recommended model')
    
    args = parser.parse_args()
    
    print("SEMA Model Installation CLI")
    print("=" * 50)
    
    installer = ModelInstaller()
    
    # Select model
    model_name, model_config = installer.select_model(args.model)
    
    if args.check_only:
        print(f"🎯 Recommended model: {model_name}")
        print(f"📋 Description: {model_config['description']}")
        print(f"🔗 Repository: {model_config['repo']}")
        return
    
    # Install dependencies
    if not args.no_deps:
        if not installer.install_dependencies():
            print("❌ Failed to install dependencies")
            sys.exit(1)
    
    # Download model
    success = installer.download_model(model_config["repo"], args.dir)
    if not success:
        sys.exit(1)
    
    # Check data files
    installer.setup_data_files()
    
    print("\n🎉 Installation complete!")
    print(f"Model: {model_name} ({model_config['repo']})")
    print(f"Location: {args.dir}")
    print("\nNext steps:")
    print("1. Ensure data files (data2.pkl, voc_etc.pkl, keyword_doc.pkl) are in ./data/")
    print("2. Run: python -m src.cli input_file.xlsx")

if __name__ == '__main__':
    main()