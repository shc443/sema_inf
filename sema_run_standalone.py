#!/usr/bin/env python3
"""
SEMA Run - Standalone Zero-Configuration CLI
=============================================

Simple command to process all VOC files with zero arguments.
Works directly without pip install.

Usage:
    python sema_run_standalone.py
"""

import os
import sys
from pathlib import Path

def main():
    """Run SEMA processing with zero configuration"""

    print("🎯 SEMA VOC Analysis")
    print("=" * 50)
    print()

    # Set default paths
    input_dir = "data/input"
    output_dir = "data/output"

    # Create directories if they don't exist
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Check if there are any files to process
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx') and not f.startswith('~')]

    if not input_files:
        print(f"❌ No Excel files found in {input_dir}/")
        print()
        print("📁 Please add your Excel files to the data/input/ folder")
        print("   Files should have VOC1 and VOC2 columns with Korean text")
        print()
        sys.exit(1)

    print(f"📂 Input:  {input_dir}/")
    print(f"📂 Output: {output_dir}/")
    print(f"📊 Found {len(input_files)} files to check")
    print()

    # Import and run the CLI with default arguments
    try:
        # Fix imports for standalone execution
        project_root = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(project_root, 'src')

        # Add src directory to Python path
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        # Import with package context to handle relative imports
        import importlib.util
        spec = importlib.util.spec_from_file_location("cli", os.path.join(src_dir, "cli.py"))
        cli_module = importlib.util.module_from_spec(spec)
        sys.modules['cli'] = cli_module
        spec.loader.exec_module(cli_module)

        SemaInference = cli_module.SemaInference

        # Initialize inference engine
        print("🔧 Loading SEMA model...")
        inferencer = SemaInference()

        # Process directory with smart skipping
        print()
        inferencer.process_directory(input_dir, output_dir, force=False)

        print()
        print("=" * 50)
        print("✅ Processing complete!")
        print(f"📁 Results saved to: {output_dir}/")

        # If in Colab, offer to download
        try:
            import google.colab
            from google.colab import files
            print()
            print("📥 Downloading results...")
            for f in os.listdir('data/output'):
                if f.endswith('.xlsx'):
                    files.download(f'data/output/{f}')
            print("✅ All files downloaded!")
        except ImportError:
            pass

    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("💡 Troubleshooting:")
        print("   1. Make sure required model files are present")
        print("   2. Check that Excel files have VOC1 and VOC2 columns")
        print("   3. Verify you have a GPU runtime in Colab")
        sys.exit(1)

if __name__ == '__main__':
    main()
