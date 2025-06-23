# Running SEMA Inference on Google Colab

This guide will help you run the SEMA inference CLI on Google Colab.

## 🚀 Quick Start

### Step 1: Setup Environment

```python
# Run this in your first Colab cell
!wget https://raw.githubusercontent.com/your-repo/sema_inf/main/colab_setup.py
exec(open('colab_setup.py').read())
run_colab_setup()
```

### Step 2: Upload Required Files

```python
# Run in second cell
upload_required_files()
```

You'll need to upload these files:
- `data2.pkl` - MultiLabelBinarizer for label encoding
- `voc_etc.pkl` - Filter data for VOC preprocessing  
- `keyword_doc.pkl` - Keyword extraction mappings
- `deberta-v3-xlarge-korean_20ep_full_mar17_dropna.ckpt` - Trained model checkpoint

### Step 3: Verify Setup

```python
# Run in third cell
verify_setup()
```

### Step 4: Clone Repository and Install

```python
# Clone the repository
!git clone https://github.com/your-username/sema_inf.git
%cd sema_inf

# Install in development mode
!pip install -e .
```

### Step 5: Run Inference

```python
from src.cli import SemaInference

# Initialize inference engine
inferencer = SemaInference(
    model_path='team-lucid/deberta-v3-xlarge-korean',
    checkpoint_path='your_model_checkpoint.ckpt'
)

# Upload your Excel file
from google.colab import files
uploaded = files.upload()

# Process the uploaded file
for filename in uploaded.keys():
    if filename.endswith('.xlsx'):
        print(f"Processing {filename}...")
        success = inferencer.process_file(filename, f"output_{filename}")
        if success:
            print("✅ Processing completed!")
            # Download results
            files.download(f"output_{filename}")
```

## 🔧 Manual Setup (Alternative)

If the automatic setup doesn't work, follow these manual steps:

### Install System Dependencies

```bash
# Install Java for KoNLPy
!apt-get update -qq
!apt-get install -y openjdk-8-jdk

# Set Java environment
import os
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64'
```

### Install Python Packages

```bash
!pip install torch transformers>=4.30.0 konlpy pandas numpy scikit-learn openpyxl tqdm torchmetrics
```

### Test KoNLPy Installation

```python
from konlpy.tag import Kkma
kkma = Kkma()
print("KoNLPy test:", kkma.morphs("한국어 테스트"))
```

## 📁 File Structure in Colab

After setup, your Colab environment should look like this:

```
/content/
├── sema_inf/
│   ├── src/
│   │   ├── cli.py
│   │   └── sema.py
│   ├── data/
│   │   ├── data2.pkl
│   │   ├── voc_etc.pkl
│   │   └── keyword_doc.pkl
│   └── your_model_checkpoint.ckpt
```

## ⚠️ Common Issues and Solutions

### 1. Java Not Found Error
```bash
# Reinstall Java
!apt-get install -y openjdk-8-jdk
import os
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64'
```

### 2. CUDA Out of Memory
```python
# Reduce batch size
inferencer.batch_size = 4  # Default is 12
```

### 3. Model Checkpoint Not Found
```python
# Check if checkpoint file exists
import os
print("Files in current directory:", os.listdir('.'))
print("Files in data directory:", os.listdir('data/'))

# Update checkpoint path
inferencer.checkpoint_path = 'path/to/your/checkpoint.ckpt'
```

### 4. Korean Text Processing Issues
```python
# Test KoNLPy installation
from konlpy.tag import Kkma
try:
    kkma = Kkma()
    print("✅ KoNLPy working")
except Exception as e:
    print(f"❌ KoNLPy error: {e}")
    # Reinstall
    !pip install --upgrade konlpy
```

## 🎯 Tips for Better Performance

1. **Use GPU Runtime**: Runtime → Change runtime type → GPU
2. **Reduce Batch Size**: If you get memory errors, reduce batch_size
3. **Process Small Files First**: Test with smaller datasets
4. **Save Intermediate Results**: Save processed data to avoid reprocessing

## 📊 Memory Management

```python
# Check GPU memory
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## 🔄 Batch Processing Multiple Files

```python
# Upload multiple files
uploaded = files.upload()

# Process all Excel files
for filename in uploaded.keys():
    if filename.endswith('.xlsx'):
        print(f"Processing {filename}...")
        success = inferencer.process_file(filename, f"output_{filename}")
        if success:
            print(f"✅ {filename} completed")
        else:
            print(f"❌ {filename} failed")

# Download all results
import glob
for output_file in glob.glob("output_*.xlsx"):
    files.download(output_file)
```

## 🆘 Troubleshooting

If you encounter issues:

1. **Restart Runtime**: Runtime → Restart runtime
2. **Check Logs**: Look for error messages in the output
3. **Verify Files**: Ensure all required files are uploaded
4. **Test Components**: Run verification steps individually

## 📞 Support

If you need help:
1. Check the error messages carefully
2. Verify all required files are present
3. Make sure Java and KoNLPy are working
4. Try with a smaller test file first