#!/bin/bash
# SEMA Docker Quick Run Script
# Usage: ./docker-run.sh [build|run|run-cpu|shell|clean]

set -e

IMAGE_NAME="sema-inference:latest"
INPUT_DIR="$(pwd)/data/input"
OUTPUT_DIR="$(pwd)/data/output"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_green() {
    echo -e "${GREEN}$1${NC}"
}

print_yellow() {
    echo -e "${YELLOW}$1${NC}"
}

print_red() {
    echo -e "${RED}$1${NC}"
}

# Build Docker image
build() {
    print_yellow "🔨 Building Docker image..."
    docker build -t $IMAGE_NAME .
    print_green "✅ Build complete!"
}

# Run with GPU
run() {
    print_yellow "🚀 Running SEMA inference with GPU..."

    # Check if input directory exists and has files
    if [ ! -d "$INPUT_DIR" ] || [ -z "$(ls -A $INPUT_DIR/*.xlsx 2>/dev/null)" ]; then
        print_red "❌ No Excel files found in $INPUT_DIR"
        print_yellow "Please add your Excel files to data/input/ first"
        exit 1
    fi

    # Create output directory if needed
    mkdir -p "$OUTPUT_DIR"

    # Run container
    docker run --rm --gpus all \
        -v "$INPUT_DIR:/workspace/data/input" \
        -v "$OUTPUT_DIR:/workspace/data/output" \
        $IMAGE_NAME

    print_green "✅ Processing complete! Check $OUTPUT_DIR for results"
}

# Run with CPU only
run_cpu() {
    print_yellow "🚀 Running SEMA inference with CPU (slower)..."

    if [ ! -d "$INPUT_DIR" ] || [ -z "$(ls -A $INPUT_DIR/*.xlsx 2>/dev/null)" ]; then
        print_red "❌ No Excel files found in $INPUT_DIR"
        print_yellow "Please add your Excel files to data/input/ first"
        exit 1
    fi

    mkdir -p "$OUTPUT_DIR"

    docker run --rm \
        -v "$INPUT_DIR:/workspace/data/input" \
        -v "$OUTPUT_DIR:/workspace/data/output" \
        $IMAGE_NAME

    print_green "✅ Processing complete! Check $OUTPUT_DIR for results"
}

# Interactive shell
shell() {
    print_yellow "🐚 Starting interactive shell..."
    docker run --rm -it --gpus all \
        -v "$INPUT_DIR:/workspace/data/input" \
        -v "$OUTPUT_DIR:/workspace/data/output" \
        $IMAGE_NAME bash
}

# Clean up
clean() {
    print_yellow "🧹 Cleaning Docker resources..."
    docker system prune -af --volumes
    print_green "✅ Cleanup complete!"
}

# Show usage
usage() {
    echo "SEMA Docker Quick Run Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build      Build Docker image"
    echo "  run        Run inference with GPU (default)"
    echo "  run-cpu    Run inference with CPU only"
    echo "  shell      Start interactive shell in container"
    echo "  clean      Clean up Docker resources"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 run"
    echo "  $0 run-cpu"
    echo ""
}

# Main
case "${1:-run}" in
    build)
        build
        ;;
    run)
        run
        ;;
    run-cpu)
        run_cpu
        ;;
    shell)
        shell
        ;;
    clean)
        clean
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        print_red "❌ Unknown command: $1"
        usage
        exit 1
        ;;
esac
