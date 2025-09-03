#!/usr/bin/env python3
"""
Colab Safe Processor - SEMA VOC Analysis with timeout and monitoring
Designed specifically for Google Colab environment
"""

import os
import sys
import time
import signal
import threading
import traceback
import json
from datetime import datetime, timedelta
from pathlib import Path
import psutil
import torch
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

class ColabSafeProcessor:
    def __init__(self, timeout_minutes=15):
        self.timeout_seconds = timeout_minutes * 60
        self.start_time = time.time()
        self.last_activity = time.time()
        self.is_processing = False
        self.current_file = ""
        self.total_files = 0
        self.processed_files = 0
        
        # Create directories
        self.setup_directories()
        
        # Setup logging for Colab
        self.setup_colab_logging()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_process, daemon=True)
        self.monitor_thread.start()
        
        print(f"🛡️ Safe processor initialized with {timeout_minutes} minute timeout per file")
    
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs('data/input', exist_ok=True)
        os.makedirs('data/output', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('logs/errors', exist_ok=True)
    
    def setup_colab_logging(self):
        """Setup logging optimized for Colab"""
        self.log_messages = []
        self.error_count = 0
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = Path(f"logs/colab_process_{timestamp}.log")
    
    def log(self, message, level="INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}"
        
        # Store in memory
        self.log_messages.append(formatted_message)
        
        # Print to Colab output
        if level == "ERROR":
            print(f"🚨 {formatted_message}")
            self.error_count += 1
        elif level == "WARNING":
            print(f"⚠️ {formatted_message}")
        else:
            print(f"📝 {formatted_message}")
        
        # Save to file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(formatted_message + '\n')
    
    def update_activity(self, message=""):
        """Update last activity timestamp"""
        self.last_activity = time.time()
        if message:
            self.log(f"Activity: {message}")
    
    def set_current_file(self, filename, file_index, total):
        """Set current processing file"""
        self.current_file = filename
        self.processed_files = file_index
        self.total_files = total
        self.update_activity(f"Processing file {file_index}/{total}: {filename}")
    
    def start_processing(self):
        """Mark start of processing"""
        self.is_processing = True
        self.update_activity("Started GPU processing")
    
    def end_processing(self):
        """Mark end of processing"""
        self.is_processing = False
        self.update_activity("Completed GPU processing")
    
    def monitor_process(self):
        """Monitor process for hangs and timeouts"""
        while True:
            time.sleep(30)  # Check every 30 seconds
            
            current_time = time.time()
            elapsed_since_activity = current_time - self.last_activity
            
            # Log system status every 2 minutes of inactivity
            if elapsed_since_activity > 120:
                self.log_system_status()
            
            # Check for timeout
            if self.is_processing and elapsed_since_activity > self.timeout_seconds:
                self.handle_timeout()
                break
    
    def log_system_status(self):
        """Log current system status"""
        try:
            # GPU status
            gpu_info = ""
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_cached = torch.cuda.memory_reserved() / 1024**3   # GB
                gpu_info = f"GPU Memory: {gpu_memory:.2f}GB allocated, {gpu_cached:.2f}GB cached"
            else:
                gpu_info = "GPU not available"
            
            # System resources
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            status_msg = (f"System Status - CPU: {cpu_percent}%, "
                         f"RAM: {memory.percent}% ({memory.available/1024**3:.1f}GB free), "
                         f"{gpu_info}")
            
            self.log(status_msg)
            
            if self.current_file:
                progress_msg = f"Current: {self.current_file} ({self.processed_files}/{self.total_files})"
                self.log(progress_msg)
                
        except Exception as e:
            self.log(f"Error logging system status: {e}", "ERROR")
    
    def handle_timeout(self):
        """Handle process timeout"""
        error_msg = f"""
🚨 PROCESS TIMEOUT DETECTED
=========================
Process has been inactive for more than {self.timeout_seconds/60} minutes.

Current Status:
- Total elapsed time: {(time.time() - self.start_time)/60:.1f} minutes
- Last activity: {(time.time() - self.last_activity)/60:.1f} minutes ago
- Current file: {self.current_file}
- Progress: {self.processed_files}/{self.total_files} files

This usually indicates:
1. GPU processing has hung
2. CUDA memory issues  
3. Model loading problems
4. System resource exhaustion

Recommended actions:
1. Restart the Colab runtime
2. Try processing files in smaller batches
3. Check if GPU is still available
"""
        
        self.log(error_msg, "ERROR")
        
        # Save detailed error report
        self.save_error_report("timeout")
        
        # Try to clear GPU memory
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.log("GPU memory cleared", "WARNING")
        except Exception as e:
            self.log(f"Failed to clear GPU memory: {e}", "ERROR")
        
        # Force stop processing
        self.is_processing = False
        raise TimeoutError("Processing timeout exceeded")
    
    def save_error_report(self, error_type):
        """Save detailed error report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = Path(f"logs/errors/colab_error_{error_type}_{timestamp}.json")
        
        try:
            error_data = {
                "timestamp": timestamp,
                "error_type": error_type,
                "elapsed_time_minutes": (time.time() - self.start_time) / 60,
                "last_activity_minutes_ago": (time.time() - self.last_activity) / 60,
                "current_file": self.current_file,
                "processed_files": self.processed_files,
                "total_files": self.total_files,
                "system_info": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "gpu_available": torch.cuda.is_available(),
                },
                "recent_logs": self.log_messages[-20:],  # Last 20 log messages
                "error_count": self.error_count
            }
            
            if torch.cuda.is_available():
                error_data["gpu_info"] = {
                    "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                    "memory_cached_gb": torch.cuda.memory_reserved() / 1024**3,
                    "device_name": torch.cuda.get_device_name(0)
                }
            
            with open(error_file, 'w') as f:
                json.dump(error_data, f, indent=2)
            
            self.log(f"Error report saved to: {error_file}")
            
        except Exception as e:
            self.log(f"Failed to save error report: {e}", "ERROR")
    
    def process_file_safely(self, sema, filename, file_index, total_files):
        """Process a single file with safety checks"""
        self.set_current_file(filename, file_index, total_files)
        
        try:
            self.start_processing()
            
            # Add heartbeat during processing
            def heartbeat():
                count = 0
                while self.is_processing:
                    time.sleep(60)  # Update every minute
                    if self.is_processing:
                        count += 1
                        self.update_activity(f"Still processing {filename} ({count} min)")
                        
                        # Show progress in Colab
                        elapsed = (time.time() - self.last_activity) / 60
                        if elapsed > 5:  # Show warning after 5 minutes
                            print(f"⏰ Processing {filename} for {elapsed:.1f} minutes...")
            
            heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
            heartbeat_thread.start()
            
            # Process the file
            self.log(f"Starting processing: {filename}")
            success = sema.process_file(filename)
            
            self.end_processing()
            
            if success:
                self.log(f"✅ Successfully completed: {filename}")
            else:
                self.log(f"❌ Failed to process: {filename}", "ERROR")
            
            return success
            
        except Exception as e:
            self.end_processing()
            error_msg = f"Error processing {filename}: {str(e)}"
            self.log(error_msg, "ERROR")
            self.log(traceback.format_exc(), "ERROR")
            
            # Save individual file error
            error_file = Path(f"logs/errors/file_error_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(error_file, 'w') as f:
                f.write(f"Error processing {filename}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Traceback:\n{traceback.format_exc()}")
            
            return False

def safe_process_all_files():
    """Main function for safe processing in Colab"""
    print("🤖 SEMA Safe Processing for Google Colab")
    print("=" * 45)
    
    # Initialize safe processor
    processor = ColabSafeProcessor(timeout_minutes=15)
    
    try:
        processor.log("Initializing SEMA CLI for Colab")
        
        # Import and initialize SEMA
        try:
            from colab_cli import SemaColabCLI
            sema = SemaColabCLI()
            processor.log("✅ SEMA CLI initialized successfully")
        except Exception as e:
            processor.log(f"❌ Error initializing SEMA: {str(e)}", "ERROR")
            return False
        
        # Check for input files
        input_files = [f for f in os.listdir('data/input') if f.endswith('.xlsx') and not f.startswith('~')]
        
        if not input_files:
            processor.log("❌ No Excel files found in data/input directory", "ERROR")
            print("\n💡 To upload files:")
            print("1. Run the upload cell above")
            print("2. Or manually copy files to data/input/")
            return False
        
        processor.log(f"📂 Found {len(input_files)} files to process")
        for i, file in enumerate(input_files, 1):
            processor.log(f"   {i}. {file}")
        
        print(f"\n🔄 Processing {len(input_files)} files with safety monitoring...")
        print(f"⏰ Timeout: 15 minutes per file")
        print(f"📊 Progress will be logged in real-time")
        print(f"🚨 Error reports saved to logs/errors/")
        
        # Process files
        success_count = 0
        failed_files = []
        
        for i, filename in enumerate(input_files, 1):
            print(f"\n📄 Processing {i}/{len(input_files)}: {filename}")
            
            try:
                if processor.process_file_safely(sema, filename, i, len(input_files)):
                    success_count += 1
                    print(f"✅ {filename} completed successfully")
                else:
                    failed_files.append(filename)
                    print(f"❌ {filename} failed")
                    
            except TimeoutError:
                print(f"🚨 {filename} timed out - stopping processing")
                failed_files.append(filename)
                break
            except Exception as e:
                print(f"🚨 Critical error with {filename}: {e}")
                failed_files.append(filename)
        
        # Final summary
        print(f"\n🎉 Processing Summary:")
        print(f"✅ Successfully processed: {success_count}/{len(input_files)} files")
        
        if failed_files:
            print(f"❌ Failed files: {len(failed_files)}")
            for failed_file in failed_files:
                print(f"   • {failed_file}")
        
        # Show results
        if success_count > 0:
            output_files = [f for f in os.listdir('data/output') if f.endswith('.xlsx')]
            print(f"\n📁 Generated {len(output_files)} output files:")
            for file in output_files:
                size = os.path.getsize(f'data/output/{file}') / 1024  # KB
                print(f"   • {file} ({size:.1f} KB)")
            
            print(f"\n✅ Processing completed!")
            print(f"📂 Results are in data/output/ folder")
            
            # Auto-download results
            try:
                from google.colab import files
                print("\n📥 Auto-downloading results...")
                for file in output_files:
                    files.download(f"data/output/{file}")
                print("✅ All files downloaded!")
            except Exception as e:
                print(f"⚠️ Auto-download failed: {e}")
                print("💡 You can manually download from data/output/ folder")
        
        processor.log(f"Processing completed: {success_count}/{len(input_files)} successful")
        return success_count > 0
        
    except Exception as e:
        processor.log(f"Critical error during processing: {e}", "ERROR")
        processor.log(traceback.format_exc(), "ERROR")
        processor.save_error_report("critical")
        
        print(f"\n🚨 Critical Error: {e}")
        print("📁 Check logs/errors/ for detailed error reports")
        return False

# Function to be called from notebook
def run_safe_processing():
    """Entry point for Colab notebook"""
    return safe_process_all_files()