#!/usr/bin/env python3
"""
SEMA VOC Analysis - Complete Processing Pipeline (Google Drive Version)
=======================================================================

This file does EVERYTHING from Google Drive:
1. Mounts Google Drive automatically
2. Sets up environment and installs dependencies
3. Finds Excel files in your Google Drive input folder
4. Processes all files with safety monitoring
5. Saves results to Google Drive output folder

Upload this file and your Excel files to Google Drive, then run it in Colab!

FOLDER STRUCTURE IN GOOGLE DRIVE:
/MyDrive/SEMA/
├── run_all.py (this file)
├── input/ (put your Excel files here)
└── output/ (results will be saved here)
"""

import os
import sys
import time
import subprocess
import threading
import traceback
import json
from datetime import datetime
from pathlib import Path

def mount_google_drive():
    """Mount Google Drive"""
    print("📱 Mounting Google Drive...")
    
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully")
        
        # Check if SEMA folder exists
        sema_path = '/content/drive/MyDrive/SEMA'
        if not os.path.exists(sema_path):
            os.makedirs(sema_path, exist_ok=True)
            print(f"📁 Created SEMA folder: {sema_path}")
        
        # Create input and output folders
        input_path = f"{sema_path}/input"
        output_path = f"{sema_path}/output"
        logs_path = f"{sema_path}/logs"
        
        os.makedirs(input_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f"{logs_path}/errors", exist_ok=True)
        
        print(f"📁 Folders ready:")
        print(f"   📂 Input: {input_path}")
        print(f"   📂 Output: {output_path}")
        print(f"   📂 Logs: {logs_path}")
        
        return sema_path
        
    except ImportError:
        print("❌ Not running in Google Colab")
        return None
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {e}")
        return None

def setup_environment():
    """Setup complete environment"""
    print("🔧 Setting up SEMA environment...")
    
    # Install system dependencies
    try:
        subprocess.run(['apt-get', 'update', '-qq'], check=True, capture_output=True)
        subprocess.run(['apt-get', 'install', '-y', '-qq', 'openjdk-8-jdk'], check=True, capture_output=True)
        print("✅ Java installed")
        
        # Set Java environment
        os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64'
        print("✅ Java environment set")
        
    except:
        print("⚠️ Java installation failed - may already be installed")
    
    # Install Python packages
    packages = [
        'huggingface_hub>=0.16.0',
        'torch>=2.0.0', 
        'transformers>=4.30.0,<5.0.0',
        'torchmetrics>=0.11.0',
        'lightning>=2.0.0',
        'konlpy',
        'psutil'
    ]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package, '-q'], check=True)
            print(f"✅ {package.split('>=')[0].split('==')[0]} installed")
        except:
            print(f"⚠️ {package} installation failed")
    
    # Clone repository if needed
    if not os.path.exists('sema_inf'):
        try:
            subprocess.run(['git', 'clone', '-q', 'https://github.com/shc443/sema_inf.git'], check=True)
            print("✅ Repository cloned")
        except:
            print("⚠️ Repository clone failed - may already exist")
    
    # Change to repo directory
    try:
        os.chdir('sema_inf')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.'], check=True)
        print("✅ SEMA package installed")
    except:
        print("⚠️ Package installation failed")
    
    # Test installations
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
        else:
            print("⚠️ No GPU - processing will be slower")
    except:
        print("⚠️ PyTorch not available")
    
    try:
        import psutil
        print(f"🖥️ System: CPU={psutil.cpu_count()} cores, RAM={psutil.virtual_memory().total/1024**3:.1f}GB")
    except:
        print("⚠️ System monitoring not available")
    
    print("✅ Environment setup complete!")

class ProcessMonitor:
    def __init__(self, drive_path, timeout_minutes=15):
        self.timeout_seconds = timeout_minutes * 60
        self.start_time = time.time()
        self.last_activity = time.time()
        self.is_processing = False
        self.current_file = ""
        self.drive_path = drive_path
        
        # Set paths
        self.input_path = f"{drive_path}/input"
        self.output_path = f"{drive_path}/output" 
        self.logs_path = f"{drive_path}/logs"
        
        # Start monitoring
        self.monitor_thread = threading.Thread(target=self.monitor_process, daemon=True)
        self.monitor_thread.start()
    
    def log(self, message):
        """Log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def update_activity(self, message=""):
        """Update activity"""
        self.last_activity = time.time()
        if message:
            self.log(message)
    
    def start_processing(self, filename):
        """Start processing a file"""
        self.is_processing = True
        self.current_file = filename
        self.update_activity(f"Started processing: {filename}")
    
    def end_processing(self):
        """End processing"""
        self.is_processing = False
        self.update_activity("Processing completed")
    
    def monitor_process(self):
        """Monitor for timeouts"""
        while True:
            time.sleep(60)  # Check every minute
            
            if self.is_processing:
                elapsed = time.time() - self.last_activity
                if elapsed > self.timeout_seconds:
                    self.log(f"🚨 TIMEOUT: {self.current_file} exceeded {self.timeout_seconds/60} minutes")
                    
                    # Save error report
                    error_file = f"{self.logs_path}/errors/timeout_{self.current_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(error_file, 'w') as f:
                        f.write(f"Timeout processing {self.current_file}\n")
                        f.write(f"Exceeded {self.timeout_seconds/60} minutes\n")
                        f.write(f"Time: {datetime.now()}\n")
                    
                    # Clear GPU memory
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            self.log("GPU memory cleared")
                    except:
                        pass
                    
                    self.is_processing = False
                    break

def find_drive_files(drive_path):
    """Find Excel files in Google Drive input folder"""
    print("🔍 Looking for Excel files in Google Drive...")
    
    input_path = f"{drive_path}/input"
    
    if not os.path.exists(input_path):
        print(f"❌ Input folder not found: {input_path}")
        print("💡 Please create the folder and upload your Excel files there")
        return []
    
    try:
        files = [f for f in os.listdir(input_path) if f.endswith('.xlsx') and not f.startswith('~')]
        
        found_files = []
        for file in files:
            full_path = os.path.join(input_path, file)
            if os.path.isfile(full_path):
                size = os.path.getsize(full_path) / 1024  # KB
                found_files.append(full_path)
                print(f"📁 Found: {file} ({size:.1f} KB)")
        
        return found_files
        
    except Exception as e:
        print(f"❌ Error reading input folder: {e}")
        return []

def setup_processing_folders(drive_path):
    """Setup processing folders in /content for faster processing"""
    print("📁 Setting up processing folders...")
    
    # Create local processing folders for faster access
    os.makedirs('/content/data/input', exist_ok=True)
    os.makedirs('/content/data/output', exist_ok=True)
    os.makedirs('/content/logs/errors', exist_ok=True)
    
    # Copy files from Drive to local for processing
    drive_input = f"{drive_path}/input"
    local_input = "/content/data/input"
    
    if not os.path.exists(drive_input):
        return []
    
    copied_files = []
    try:
        import shutil
        files = [f for f in os.listdir(drive_input) if f.endswith('.xlsx') and not f.startswith('~')]
        
        for file in files:
            src = os.path.join(drive_input, file)
            dst = os.path.join(local_input, file)
            shutil.copy2(src, dst)
            copied_files.append(file)
            size = os.path.getsize(dst) / 1024
            print(f"✅ Copied to processing: {file} ({size:.1f} KB)")
            
    except Exception as e:
        print(f"❌ Error copying files: {e}")
    
    return copied_files

def process_all_files(drive_path):
    """Process all files with safety monitoring"""
    print("\n🚀 Starting SEMA processing...")
    
    monitor = ProcessMonitor(drive_path)
    
    try:
        # Import SEMA components
        from colab_cli import SemaColabCLI
        
        print("🔧 Initializing SEMA...")
        sema = SemaColabCLI()
        print("✅ SEMA initialized")
        
        # Get input files from local processing folder
        input_files = [f for f in os.listdir('/content/data/input') if f.endswith('.xlsx') and not f.startswith('~')]
        
        if not input_files:
            print("❌ No files found in processing directory")
            return False
        
        print(f"📂 Processing {len(input_files)} files:")
        for i, file in enumerate(input_files, 1):
            print(f"   {i}. {file}")
        
        # Process each file
        success_count = 0
        for i, filename in enumerate(input_files, 1):
            print(f"\n📄 Processing {i}/{len(input_files)}: {filename}")
            
            monitor.start_processing(filename)
            
            try:
                # Add heartbeat for long processing
                def heartbeat():
                    count = 0
                    while monitor.is_processing and monitor.current_file == filename:
                        time.sleep(60)
                        if monitor.is_processing:
                            count += 1
                            monitor.update_activity(f"Still processing {filename} ({count} min)")
                
                heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
                heartbeat_thread.start()
                
                # Process the file
                success = sema.process_file(filename)
                
                if success:
                    success_count += 1
                    print(f"✅ {filename} completed successfully")
                else:
                    print(f"❌ {filename} failed")
                
            except Exception as e:
                print(f"🚨 Error processing {filename}: {e}")
                
                # Save error
                error_file = f"/content/logs/errors/process_error_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(error_file, 'w') as f:
                    f.write(f"Error processing {filename}\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Traceback:\n{traceback.format_exc()}")
            
            finally:
                monitor.end_processing()
        
        # Show results
        print(f"\n🎉 Processing Summary:")
        print(f"✅ Successfully processed: {success_count}/{len(input_files)} files")
        
        if success_count > 0:
            # Copy results back to Google Drive
            copy_results_to_drive(drive_path)
            
            output_files = [f for f in os.listdir('/content/data/output') if f.endswith('.xlsx')]
            print(f"📁 Generated {len(output_files)} output files:")
            for file in output_files:
                size = os.path.getsize(f'/content/data/output/{file}') / 1024
                print(f"   • {file} ({size:.1f} KB)")
            
            return True
        
        return False
        
    except Exception as e:
        print(f"🚨 Critical error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Save critical error
        error_file = f"/content/logs/errors/critical_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(error_file, 'w') as f:
            f.write(f"Critical error\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}")
        
        return False

def copy_results_to_drive(drive_path):
    """Copy results back to Google Drive"""
    print("\n📁 Saving results to Google Drive...")
    
    try:
        import shutil
        
        local_output = "/content/data/output"
        drive_output = f"{drive_path}/output"
        
        if not os.path.exists(local_output):
            print("❌ No local output folder found")
            return
            
        output_files = [f for f in os.listdir(local_output) if f.endswith('.xlsx')]
        
        if not output_files:
            print("❌ No output files to save")
            return
        
        print(f"💾 Saving {len(output_files)} files to Google Drive...")
        
        for file in output_files:
            src = os.path.join(local_output, file)
            dst = os.path.join(drive_output, file)
            shutil.copy2(src, dst)
            size = os.path.getsize(dst) / 1024
            print(f"✅ Saved to Drive: {file} ({size:.1f} KB)")
        
        # Also copy error logs if any
        local_logs = "/content/logs"
        drive_logs = f"{drive_path}/logs"
        
        if os.path.exists(local_logs):
            try:
                if not os.path.exists(drive_logs):
                    os.makedirs(drive_logs, exist_ok=True)
                    
                for root, dirs, files in os.walk(local_logs):
                    for file in files:
                        if file.endswith('.txt') or file.endswith('.json'):
                            src = os.path.join(root, file)
                            # Recreate directory structure
                            rel_path = os.path.relpath(root, local_logs)
                            dst_dir = os.path.join(drive_logs, rel_path)
                            os.makedirs(dst_dir, exist_ok=True)
                            dst = os.path.join(dst_dir, file)
                            shutil.copy2(src, dst)
                            
                print("✅ Log files saved to Google Drive")
            except Exception as e:
                print(f"⚠️ Could not save logs: {e}")
        
        print(f"🎉 All results saved to: {drive_output}")
        
    except Exception as e:
        print(f"❌ Failed to save to Google Drive: {e}")

def download_results_from_drive(drive_path):
    """Download results from Google Drive to local downloads"""
    print("\n📥 Downloading results from Google Drive...")
    
    try:
        from google.colab import files
        
        drive_output = f"{drive_path}/output"
        
        if not os.path.exists(drive_output):
            print("❌ No output folder found in Google Drive")
            return
            
        output_files = [f for f in os.listdir(drive_output) if f.endswith('.xlsx')]
        
        if not output_files:
            print("❌ No output files to download")
            return
        
        print(f"📁 Downloading {len(output_files)} files from Google Drive...")
        for file in output_files:
            file_path = os.path.join(drive_output, file)
            files.download(file_path)
            print(f"✅ Downloaded: {file}")
        
        print("🎉 All files downloaded successfully!")
        
    except ImportError:
        print("⚠️ Not in Colab environment")
        print(f"📂 Files are saved in: {drive_path}/output")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print(f"📂 Files are still saved in: {drive_path}/output")

def main():
    """Main execution function"""
    print("🎯 SEMA VOC Analysis - Google Drive Pipeline")
    print("=" * 50)
    print()
    
    try:
        # 1. Mount Google Drive
        drive_path = mount_google_drive()
        
        if not drive_path:
            print("❌ Failed to mount Google Drive")
            print("💡 Make sure you're running this in Google Colab")
            return
        
        print()
        
        # 2. Setup environment
        setup_environment()
        print()
        
        # 3. Find files in Google Drive
        drive_files = find_drive_files(drive_path)
        
        if not drive_files:
            print("❌ No Excel files found in Google Drive!")
            print(f"💡 Please upload Excel files to: {drive_path}/input/")
            print("📂 Create the 'input' folder in your SEMA directory and upload files there")
            return
        
        print(f"✅ Found {len(drive_files)} Excel files in Google Drive")
        print()
        
        # 4. Setup processing folders and copy files
        copied_files = setup_processing_folders(drive_path)
        
        if not copied_files:
            print("❌ No files copied for processing")
            return
        
        print(f"✅ {len(copied_files)} files ready for processing")
        print()
        
        # 5. Process all files
        success = process_all_files(drive_path)
        
        if not success:
            print("\n❌ Processing failed")
            print(f"🔍 Check {drive_path}/logs/errors/ for detailed error reports")
            return
        
        # 6. Download results (optional)
        print("\n🎉 PROCESSING COMPLETE!")
        print(f"💾 Results saved to Google Drive: {drive_path}/output/")
        
        download_choice = input("\n📥 Do you want to download results to your computer? (y/n): ")
        if download_choice.lower() == 'y':
            download_results_from_drive(drive_path)
        
        print("\n🎉 ALL DONE!")
        print(f"📂 Your results are saved in Google Drive: {drive_path}/output/")
        print("✅ Processing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n🚨 Fatal error: {e}")
        print("📁 Check Google Drive logs/errors/ for detailed error reports")

if __name__ == "__main__":
    main()