#!/usr/bin/env python3
"""
Simple Google Colab CLI for SEMA inference
Designed for client use with file upload interface
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import re
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoConfig
from konlpy.tag import Kkma
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from google.colab import files
import shutil

class SemaColabCLI:
    def __init__(self):
        self.model_name = 'team-lucid/deberta-v3-xlarge-korean'
        self.hf_repo = 'shc443/sema2025'  # Your HF repository
        self.max_len = 256
        self.batch_size = 8  # Reduced for Colab
        self.opt_thresh = 0.5
        
        print("🚀 Initializing SEMA CLI for Google Colab...")
        self._setup_directories()
        self._setup_environment()
        self._load_components()
        
    def _setup_directories(self):
        """Create necessary directories"""
        os.makedirs('data', exist_ok=True)
        os.makedirs('data/input', exist_ok=True)
        os.makedirs('data/output', exist_ok=True)
        os.makedirs('model', exist_ok=True)
        print("✅ Directories created")
    
    def _setup_environment(self):
        """Setup environment for Colab"""
        try:
            # Check if we're in Colab
            import google.colab
            print("✅ Running in Google Colab")
            
            # Set Java environment for KoNLPy
            os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64'
            
            # Install and test KoNLPy
            import subprocess
            import sys
            print("📦 Installing KoNLPy...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'konlpy', '-q'], check=True)
            
            # Test KoNLPy installation
            try:
                from konlpy.tag import Kkma
                kkma = Kkma()
                test_result = kkma.morphs("테스트")
                print("✅ KoNLPy working correctly")
            except Exception as e:
                print(f"⚠️ KoNLPy test failed: {e}")
                print("Attempting to fix...")
                # Try to reinstall
                subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'konlpy', '-q'], check=True)
            
        except ImportError:
            print("⚠️ Not running in Google Colab - some features may not work")
    
    def _download_from_hf(self, filename, subfolder=None):
        """Download files from Hugging Face repository"""
        try:
            print(f"📥 Downloading {filename} from {self.hf_repo}...")
            
            local_path = hf_hub_download(
                repo_id=self.hf_repo,
                filename=filename,
                subfolder=subfolder,
                cache_dir="./cache"
            )
            print(f"✅ Downloaded {filename}")
            return local_path
            
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            return None
    
    def _ensure_file_exists(self, filepath, hf_filename=None, subfolder=None):
        """Ensure file exists locally, download from HF if needed"""
        if os.path.exists(filepath):
            return filepath
            
        if hf_filename:
            print(f"📁 {filepath} not found locally, downloading from Hugging Face...")
            downloaded_path = self._download_from_hf(hf_filename, subfolder)
            if downloaded_path:
                # Copy to expected location
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                shutil.copy2(downloaded_path, filepath)
                return filepath
        
        return None
    
    def _load_components(self):
        """Load all model components"""
        print("🔧 Loading model components...")
        
        try:
            # Load tokenizer and config
            self.config = AutoConfig.from_pretrained(self.model_name, output_hidden_states=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("✅ Tokenizer loaded")
            
            # Load data files with auto-download from HF
            data2_path = self._ensure_file_exists('data/data2.pkl', 'data2.pkl')
            if not data2_path:
                raise FileNotFoundError("data2.pkl not found")
                
            with open(data2_path, 'rb') as f:
                self.mlb = pickle.load(f)
            print("✅ Label encoder loaded")
            
            self.label_columns = self.mlb.classes_[:]
            
            voc_etc_path = self._ensure_file_exists('data/voc_etc.pkl', 'voc_etc.pkl')
            if not voc_etc_path:
                raise FileNotFoundError("voc_etc.pkl not found")
                
            with open(voc_etc_path, 'rb') as f:
                self.voc_etc = pickle.load(f)
            print("✅ Filter data loaded")
                
            keyword_path = self._ensure_file_exists('data/keyword_doc.pkl', 'keyword_doc.pkl')
            if not keyword_path:
                raise FileNotFoundError("keyword_doc.pkl not found")
                
            with open(keyword_path, 'rb') as f:
                self.keyword = pickle.load(f)
            print("✅ Keywords loaded")
            
            # Load model checkpoint
            checkpoint_filename = "deberta-v3-xlarge-korean_20ep_full_mar17_dropna.ckpt"
            checkpoint_path = self._ensure_file_exists(f'model/{checkpoint_filename}', checkpoint_filename)
            if not checkpoint_path:
                raise FileNotFoundError(f"Model checkpoint {checkpoint_filename} not found")
            
            # Import here to avoid circular imports
            sys.path.append('.')
            from src.sema import VOC_TopicLabeler
            
            self.model = VOC_TopicLabeler.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                n_classes=len(self.label_columns),
                model=self.model_name
            )
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                print("✅ Model moved to GPU")
            else:
                print("⚠️ GPU not available, using CPU")
                
            self.model.eval()
            print("✅ Model loaded and ready")
            
            # Initialize Korean morphological analyzer
            try:
                from konlpy.tag import Kkma
                self.kkma = Kkma()
                # Test with a simple Korean text
                test_result = self.kkma.morphs("테스트")
                print("✅ Korean language processor ready")
            except Exception as e:
                print(f"⚠️ KoNLPy initialization failed: {e}")
                print("Attempting to reinstall and retry...")
                import subprocess
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'konlpy', '-q'], check=True)
                from konlpy.tag import Kkma
                self.kkma = Kkma()
                print("✅ Korean language processor ready (after reinstall)")
            
            print(f"🎉 All components loaded! Ready to process with {len(self.label_columns)} classes")
            
        except Exception as e:
            print(f"❌ Error loading components: {e}")
            raise
    
    def upload_files(self):
        """Upload files through Colab interface"""
        print("📤 Please upload your Excel files for processing:")
        print("Multiple files can be selected at once")
        
        uploaded = files.upload()
        
        if not uploaded:
            print("❌ No files uploaded")
            return []
        
        # Move uploaded files to input directory
        uploaded_files = []
        for filename in uploaded.keys():
            if filename.endswith('.xlsx') and not filename.startswith('~'):
                input_path = f"data/input/{filename}"
                shutil.move(filename, input_path)
                uploaded_files.append(filename)
                print(f"✅ Moved {filename} to data/input/")
            else:
                print(f"⚠️ Skipped {filename} (not an Excel file)")
        
        return uploaded_files
    
    def _remove_non_english_korean(self, string):
        """Remove non-English/Korean characters"""
        pattern = re.compile(r'[^a-zA-Z0-9\uac00-\ud7a3\s]', flags=re.UNICODE)
        return pattern.sub('', string)
    
    def _strip_emoji(self, text):
        """Remove emojis"""
        RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
        return RE_EMOJI.sub(r'', text)
    
    def _findall_vec(self, key, voc):
        """Extract keywords using regex"""
        try:
            return re.findall(key, voc)[0]
        except:
            return ''
    
    def _findall_vec2(self, df):
        """Apply keyword extraction to dataframe"""
        return self._findall_vec(df['keyword'], df['VOC'])
    
    def _filter_voc_data(self, voc_data):
        """Filter and clean VOC data"""
        print("🧹 Filtering VOC data...")
        
        # Basic cleaning
        voc_data = voc_data[voc_data['VOC'] != 'nan']
        voc_data['VOC'] = voc_data['VOC'].apply(self._remove_non_english_korean)
        voc_data['VOC'] = voc_data['VOC'].apply(self._strip_emoji)
        voc_data['VOC'] = voc_data['VOC'].replace(r'\s+', ' ', regex=True)
        
        # Apply filters
        filt0 = (voc_data['VOC'].str.strip().str.len() < 4).astype(int)
        filt1 = voc_data['VOC'].apply(lambda x: bool(re.match(r'^[_\W]+$', str(x).replace(' ', '')))).astype(int)
        filt2 = voc_data['VOC'].apply(lambda x: bool(re.match(r'[\d/-]+$', str(x).replace(' ', '')))).astype(int)
        filt3 = (voc_data.VOC.str.replace(' ', '').str.split('').apply(set).str.len() == 2)
        
        voc_data = voc_data[(filt0 + filt1 + filt2 + filt3) == 0]
        
        # Morphological analysis filtering
        voc_tok = voc_data['VOC'].progress_apply(lambda x: Counter(self.kkma.morphs(x)))
        filt4 = voc_tok.isin(self.voc_etc).astype(int)
        voc_data = voc_data[~filt4.astype(bool)].reset_index()
        
        print(f"✅ Filtered to {len(voc_data)} valid entries")
        return voc_data
    
    def process_file(self, filename):
        """Process a single Excel file"""
        input_path = f"data/input/{filename}"
        output_path = f"data/output/{filename.replace('.xlsx', '_output.xlsx')}"
        
        if not os.path.exists(input_path):
            print(f"❌ File not found: {input_path}")
            return False
            
        print(f"🔄 Processing: {filename}")
        
        try:
            # Read input file
            voc_testset = pd.read_excel(input_path, dtype=str)
            
            # Process VOC columns
            voc = pd.concat([voc_testset.VOC1, voc_testset.VOC2]).sort_index().values
            voc_testset = pd.concat([voc_testset]*2).sort_index().iloc[:, 1:-2]
            voc_testset['VOC'] = voc
            voc_testset = voc_testset.dropna(subset='VOC')
            voc_testset.reset_index(inplace=True)
            voc_testset['label'] = pd.DataFrame(np.zeros((len(self.label_columns), voc_testset.shape[0])).T).astype(int).apply(list, axis=1)
            
            # Filter data
            voc_testset = self._filter_voc_data(voc_testset)
            
            if len(voc_testset) == 0:
                print("❌ No valid data remaining after filtering")
                return False
            
            # Setup data module for inference
            print("🧠 Running inference...")
            sys.path.append('.')
            from src.sema import VOC_DataModule
            
            data_module = VOC_DataModule(voc_testset, voc_testset, self.tokenizer, 
                                       batch_size=self.batch_size, max_token_len=self.max_len)
            data_module.setup()
            
            # Run inference
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            predictions = self.model.predict(data_module.predict_dataloader(), device=device)
            sema_predictions = np.vstack(predictions)
            pred_labels = (sema_predictions > self.opt_thresh).astype(int)
            
            # Process results
            voc_testset['pred'] = pd.Series(self.mlb.inverse_transform(pred_labels)).apply(list)
            voc_testset = voc_testset.explode('pred', ignore_index=True)
            
            del voc_testset['label']
            
            # Extract keywords
            print("🔍 Extracting keywords...")
            voc_testset['topic'] = voc_testset.pred.str.split('_').str[0]
            voc_testset['sentiment'] = voc_testset.pred.str.split('_').str[1]
            voc_testset['topic'] = voc_testset['topic'].fillna('기타')
            voc_testset['keyword'] = self.keyword.loc[voc_testset.topic].values
            voc_testset['keyword'] = voc_testset.apply(self._findall_vec2, axis=1)
            
            # Save output
            voc_testset.to_excel(output_path, index=False)
            print(f"✅ Results saved to: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            return False
    
    def process_all_files(self):
        """Process all files in input directory, skipping already processed ones"""
        input_files = [f for f in os.listdir('data/input') if f.endswith('.xlsx') and not f.startswith('~')]

        if not input_files:
            print("❌ No Excel files found in data/input directory")
            return 0

        # Check which files are already processed
        output_files = set()
        if os.path.exists('data/output'):
            for f in os.listdir('data/output'):
                if f.endswith('_output.xlsx'):
                    # Extract original filename
                    original = f.replace('_output.xlsx', '.xlsx')
                    output_files.add(original)

        # Filter to only unprocessed files
        unprocessed = []
        already_done = []
        for filename in input_files:
            if filename in output_files:
                already_done.append(filename)
            else:
                unprocessed.append(filename)

        print(f"📊 Found {len(input_files)} total files in data/input/")
        if already_done:
            print(f"⏭️  Skipping {len(already_done)} already processed files")

        if not unprocessed:
            print("✅ All files already processed!")
            return 0

        print(f"🔄 Processing {len(unprocessed)} new files...")

        success_count = 0
        for filename in unprocessed:
            if self.process_file(filename):
                success_count += 1

        print(f"🎉 Successfully processed {success_count}/{len(unprocessed)} new files")
        return success_count
    
    def download_results(self):
        """Download all output files"""
        output_files = [f for f in os.listdir('data/output') if f.endswith('.xlsx')]
        
        if not output_files:
            print("❌ No output files found")
            return
        
        print(f"📥 Downloading {len(output_files)} result files...")
        
        for filename in output_files:
            files.download(f"data/output/{filename}")
        
        print("✅ All files downloaded!")
    
    def run_interactive(self):
        """Run interactive CLI"""
        print("""
🎯 SEMA VOC Analysis - Google Colab CLI
=====================================

Available commands:
1. Upload files and process
2. Process existing files in data/input
3. Download results
4. Show status
5. Exit
        """)
        
        while True:
            try:
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == '1':
                    uploaded_files = self.upload_files()
                    if uploaded_files:
                        print(f"\n📊 Processing {len(uploaded_files)} uploaded files...")
                        success_count = self.process_all_files()
                        if success_count > 0:
                            self.download_results()
                
                elif choice == '2':
                    self.process_all_files()
                    self.download_results()
                
                elif choice == '3':
                    self.download_results()
                
                elif choice == '4':
                    input_files = [f for f in os.listdir('data/input') if f.endswith('.xlsx')]
                    output_files = [f for f in os.listdir('data/output') if f.endswith('.xlsx')]
                    print(f"📁 Input files: {len(input_files)}")
                    print(f"📁 Output files: {len(output_files)}")
                    print(f"🖥️ GPU available: {torch.cuda.is_available()}")
                
                elif choice == '5':
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid choice. Please enter 1-5.")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def setup_colab():
    """Setup Google Colab environment"""
    print("🔧 Setting up Google Colab environment...")
    
    # Install system dependencies
    import subprocess
    import sys
    try:
        subprocess.run(['apt-get', 'update', '-qq'], check=True)
        subprocess.run(['apt-get', 'install', '-y', '-qq', 'openjdk-8-jdk'], check=True)
        print("✅ Java installed")
        
        # Set Java environment immediately
        os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64'
        print("✅ Java environment set")
        
    except:
        print("⚠️ Java installation failed - may already be installed")
    
    # Install Python packages with compatible versions
    packages = [
        'huggingface_hub>=0.16.0',
        'torch>=2.0.0',
        'transformers>=4.30.0,<5.0.0',
        'torchmetrics>=0.11.0',
        'lightning>=2.0.0'
    ]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package, '-q'], check=True)
            print(f"✅ {package} installed")
        except:
            print(f"⚠️ {package} installation failed")
    
    # Install and test KoNLPy separately with proper testing
    print("📦 Installing KoNLPy...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'konlpy', '-q'], check=True)
        
        # Test KoNLPy installation
        from konlpy.tag import Kkma
        kkma = Kkma()
        test_result = kkma.morphs("테스트")
        print("✅ KoNLPy installed and working")
        
    except Exception as e:
        print(f"⚠️ KoNLPy installation failed: {e}")
        print("Retrying KoNLPy installation...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'konlpy', '-q'], check=True)
            from konlpy.tag import Kkma
            kkma = Kkma()
            test_result = kkma.morphs("테스트")
            print("✅ KoNLPy installed and working (after retry)")
        except Exception as e2:
            print(f"❌ KoNLPy still failing: {e2}")
    
    print("🎉 Setup complete!")

def main():
    """Main function for Colab"""
    print("""
🤗 SEMA VOC Analysis - Google Colab CLI
======================================

This tool processes Korean Voice of Customer (VOC) data using AI.

Steps:
1. Upload your Excel files (with VOC1, VOC2 columns)
2. The AI will analyze sentiment and topics
3. Download the processed results

Let's get started!
    """)
    
    try:
        # Initialize CLI
        cli = SemaColabCLI()
        
        # Run interactive mode
        cli.run_interactive()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your files and try again.")

if __name__ == "__main__":
    # If running in Colab, show setup instructions
    try:
        import google.colab
        print("Run this first:")
        print("setup_colab()")
        print("\nThen run:")
        print("main()")
    except ImportError:
        main()