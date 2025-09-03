#!/bin/bash

# SEMA VOC Analysis - Safe Automated Processing Script
# This script automates the entire VOC analysis pipeline with comprehensive safety features

set -e  # Exit on any error

echo "🎯 SEMA VOC Analysis - Safe Automated Processing"
echo "==============================================="

# Configuration
INPUT_DIR="data/input"
OUTPUT_DIR="data/output"
LOG_DIR="logs"
ERROR_DIR="logs/errors"
PYTHON_SCRIPT="process_batch_safe.py"
TIMEOUT_MINUTES=15

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Safety functions
cleanup() {
    echo ""
    echo -e "${YELLOW}🧹 Cleaning up...${NC}"
    
    # Kill any hanging Python processes
    pkill -f "process_batch_safe.py" 2>/dev/null || true
    
    # Clear GPU memory if available
    python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
    
    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Set trap for cleanup on exit
trap cleanup EXIT INT TERM

# Create directories
echo "📁 Setting up directories..."
mkdir -p "$INPUT_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$ERROR_DIR"
mkdir -p "model"

# System checks
echo "🔍 Performing system checks..."

# Check Python and required packages
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found${NC}"
    exit 1
fi

# Check for required Python packages
python3 -c "import torch, transformers, psutil" 2>/dev/null || {
    echo -e "${RED}❌ Required Python packages missing${NC}"
    echo "Please install: pip install torch transformers psutil"
    exit 1
}

# Check GPU status
echo "🖥️ GPU Status:"
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU available: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
else:
    print('⚠️ No GPU available - processing will be slower')
"

# Check if input files exist
if [ ! "$(ls -A $INPUT_DIR/*.xlsx 2>/dev/null)" ]; then
    echo -e "${RED}❌ No Excel files found in $INPUT_DIR/${NC}"
    echo "Please copy your Excel files to $INPUT_DIR/ and run again."
    echo "Example: cp /path/to/your/files/*.xlsx $INPUT_DIR/"
    exit 1
fi

# Count input files
input_count=$(ls -1 "$INPUT_DIR"/*.xlsx 2>/dev/null | wc -l)
echo "📊 Found $input_count Excel files to process"

# List input files
echo "📋 Input files:"
for file in "$INPUT_DIR"/*.xlsx; do
    filename=$(basename "$file")
    size=$(du -h "$file" | cut -f1)
    echo "   • $filename ($size)"
done

echo ""
echo -e "${BLUE}🚀 Starting safe automated processing with ${TIMEOUT_MINUTES} minute timeout...${NC}"
echo -e "${YELLOW}⏰ Process will be monitored and automatically terminated if it hangs${NC}"
echo -e "${YELLOW}📊 Progress will be logged in real-time${NC}"
echo -e "${YELLOW}🚨 Error reports will be saved to $ERROR_DIR${NC}"

# Ensure safe Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}❌ Safe processing script not found: $PYTHON_SCRIPT${NC}"
    echo "This script should be created automatically. Please check the repository."
    exit 1
fi

# Run the safe Python processing script with timeout monitoring
echo -e "${BLUE}🐍 Running safe Python processing script...${NC}"
start_time=$(date +%s)

# Run with timeout (using timeout command if available)
if command -v timeout &> /dev/null; then
    echo -e "${YELLOW}⏰ Using system timeout: $((TIMEOUT_MINUTES + 5)) minutes${NC}"
    timeout $((TIMEOUT_MINUTES * 60 + 300)) python3 "$PYTHON_SCRIPT"
    exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        echo -e "${RED}🚨 SYSTEM TIMEOUT: Process exceeded maximum time limit${NC}"
        echo -e "${YELLOW}Check logs in $LOG_DIR for details${NC}"
        exit_code=1
    fi
else
    # Fallback: run without system timeout (internal timeout still works)
    python3 "$PYTHON_SCRIPT"
    exit_code=$?
fi

# Calculate processing time
end_time=$(date +%s)
processing_time=$((end_time - start_time))
processing_minutes=$((processing_time / 60))
processing_seconds=$((processing_time % 60))

# Check results
if [ $exit_code -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 SAFE AUTOMATED PROCESSING COMPLETE!${NC}"
    echo "======================================"
    echo -e "${BLUE}⏱️ Total processing time: ${processing_minutes}m ${processing_seconds}s${NC}"
    
    # Count output files
    if [ -d "$OUTPUT_DIR" ]; then
        output_count=$(ls -1 "$OUTPUT_DIR"/*.xlsx 2>/dev/null | wc -l || echo "0")
        echo -e "${GREEN}📊 Generated $output_count output files${NC}"
        
        if [ $output_count -gt 0 ]; then
            echo "📋 Output files:"
            for file in "$OUTPUT_DIR"/*.xlsx; do
                filename=$(basename "$file")
                size=$(du -h "$file" | cut -f1)
                echo -e "   ${GREEN}• $filename ($size)${NC}"
            done
            
            echo ""
            echo -e "${GREEN}📂 Output directory: $(pwd)/$OUTPUT_DIR${NC}"
            echo -e "${GREEN}✅ All files processed successfully!${NC}"
        fi
    fi
    
    # Show log information
    echo ""
    echo -e "${BLUE}📊 Processing Information:${NC}"
    echo -e "${BLUE}📁 Logs saved to: $LOG_DIR${NC}"
    if [ -d "$ERROR_DIR" ] && [ "$(ls -A $ERROR_DIR 2>/dev/null)" ]; then
        echo -e "${YELLOW}🚨 Error reports available in: $ERROR_DIR${NC}"
    fi
    
else
    echo ""
    echo -e "${RED}❌ PROCESSING FAILED${NC}"
    echo "==================="
    echo -e "${YELLOW}⏱️ Processing time: ${processing_minutes}m ${processing_seconds}s${NC}"
    echo ""
    echo -e "${YELLOW}🔍 Troubleshooting steps:${NC}"
    echo "1. Check error logs in: $LOG_DIR"
    echo "2. Check error reports in: $ERROR_DIR"
    echo "3. Verify Excel files have VOC1 and VOC2 columns with Korean text"
    echo "4. Ensure sufficient GPU memory is available"
    echo "5. Try processing fewer files at once"
    
    # Show recent error files
    if [ -d "$ERROR_DIR" ] && [ "$(ls -A $ERROR_DIR 2>/dev/null)" ]; then
        echo ""
        echo -e "${RED}🚨 Recent error reports:${NC}"
        ls -lt "$ERROR_DIR"/*.json 2>/dev/null | head -3 | while read line; do
            echo "   $line"
        done
    fi
fi

echo ""
echo -e "${BLUE}🏁 Script completed with exit code: $exit_code${NC}"