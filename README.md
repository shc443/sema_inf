# SEMA Inference CLI

A command-line interface for running SEMA ML model inference on Voice of Customer (VOC) data.

## Features

- Process individual Excel files or entire directories
- Multi-label classification for VOC topic and sentiment analysis
- Automatic data filtering and preprocessing
- Keyword extraction from classified topics
- GPU acceleration support
- Configurable batch processing

## Installation

### Option 1: Install from source
```bash
git clone <repository-url>
cd sema_inf
pip install -r requirements.txt
pip install -e .
```

### Option 2: Direct installation
```bash
pip install -e .
```

## Required Files

Make sure the following model files are in your directories:
- `data/data2.pkl` - MultiLabelBinarizer for label encoding
- `data/voc_etc.pkl` - Filter data for VOC preprocessing
- `data/keyword_doc.pkl` - Keyword extraction mappings
- `model/deberta-v3-xlarge-korean_20ep_full_mar17_dropna.ckpt` - Model checkpoint

## Usage

### Simplest Usage (Recommended)

After installation, just put your Excel files in `data/input/` and run:
```bash
sema
```

That's it! The command will:
- ✅ Automatically process all Excel files in `data/input/`
- ✅ Save results to `data/output/` with `_output.xlsx` suffix
- ✅ Skip already processed files
- ✅ Show progress and results

**Perfect for Google Colab:**
```python
# Cell 1: Setup
!pip install -e .

# Cell 2: Process everything
!sema
```

### Standard CLI Usage

Process a single Excel file:
```bash
sema-cli input_file.xlsx -o output_file.xlsx
```

Process a directory of Excel files:
```bash
sema-cli data/input/ -o data/output/
```

### Advanced Options

```bash
sema-cli input.xlsx \
    --output output.xlsx \
    --model team-lucid/deberta-v3-xlarge-korean \
    --checkpoint custom_model.ckpt \
    --batch-size 16 \
    --threshold 0.6 \
    --max-length 512
```

## Command Line Arguments

- **input**: Input Excel file or directory path (required)
- **-o, --output**: Output file or directory path (optional)
- **-m, --model**: Model name or path (default: team-lucid/deberta-v3-xlarge-korean)
- **-c, --checkpoint**: Model checkpoint path (optional)
- **-b, --batch-size**: Batch size for inference (default: 12)
- **-t, --threshold**: Classification threshold (default: 0.5)
- **--max-length**: Maximum token length (default: 256)

## Input File Format

Input Excel files should contain:
- `VOC1` column: First VOC text
- `VOC2` column: Second VOC text (optional)

## Output Format

Output Excel files will contain:
- Original input columns
- `VOC`: Processed VOC text
- `pred`: Predicted labels
- `topic`: Extracted topic
- `sentiment`: Extracted sentiment
- `keyword`: Extracted keywords

## Examples

### Process single file with custom output
```bash
python cli.py hospital_voc_data.xlsx -o analyzed_results.xlsx
```

### Batch process directory
```bash
python cli.py ./input_data/ -o ./results/ --batch-size 8
```

### High-precision analysis
```bash
python cli.py data.xlsx --threshold 0.8 --max-length 512
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Lightning 2.0+
- Transformers 4.30+
- CUDA-capable GPU (recommended)

## Model Information

This CLI uses the team-lucid/deberta-v3-xlarge-korean model fine-tuned for Korean VOC analysis with multi-label classification capabilities.

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size with `--batch-size 4`
2. **Missing model files**: Ensure all required .pkl files are in the working directory
3. **Korean text processing issues**: Install konlpy dependencies: `pip install konlpy`