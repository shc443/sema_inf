#!/usr/bin/env python3
"""
SEMA Simple Runner - Ultra Simple for Colab
============================================

Just run: python run_simple.py

Processes all files in data/input/ and saves to data/output/
"""

import os
import sys

def main():
    print("🎯 SEMA VOC Analysis - Simple Runner")
    print("=" * 50)
    print()

    # Check directories
    if not os.path.exists('data/input'):
        print("❌ data/input/ folder not found!")
        print("Please create it and add your Excel files")
        sys.exit(1)

    input_files = [f for f in os.listdir('data/input') if f.endswith('.xlsx') and not f.startswith('~')]

    if not input_files:
        print("❌ No Excel files found in data/input/")
        print("Please add your Excel files with VOC1 and VOC2 columns")
        sys.exit(1)

    print(f"📂 Found {len(input_files)} Excel files in data/input/")
    print()

    # Use the colab_cli which handles imports correctly
    try:
        from colab_cli import SemaColabCLI

        print("🔧 Initializing SEMA...")
        sema = SemaColabCLI()

        print()
        print("🚀 Processing files...")
        success_count = sema.process_all_files()

        if success_count > 0:
            print()
            print("=" * 50)
            print(f"✅ Successfully processed {success_count} files!")
            print(f"📁 Results saved to: data/output/")

            # If in Colab, download results
            try:
                import google.colab
                print()
                print("📥 Downloading results...")
                sema.download_results()
            except ImportError:
                pass
        else:
            print()
            print("❌ No files were processed successfully")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("💡 Troubleshooting:")
        print("   1. Run: pip install -r requirements.txt")
        print("   2. Make sure model files are accessible")
        print("   3. Check Excel files have VOC1 and VOC2 columns")
        sys.exit(1)

if __name__ == '__main__':
    main()
