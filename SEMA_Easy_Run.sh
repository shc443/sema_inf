#!/bin/bash

# SEMA VOC Analysis - Easy Run (Linux/Mac)
echo "🎯 SEMA VOC Analysis - Easy Run"
echo "==================================="
echo

echo "Starting SEMA processing application..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed"
    echo "Please install Python3 first"
    echo
    read -p "Press Enter to exit..."
    exit 1
fi

# Run the Python GUI application
python3 SEMA_Easy_Run.py

echo
echo "Press Enter to exit..."
read