#!/usr/bin/env python3
"""
Safe Batch processing script for SEMA VOC Analysis with timeout monitoring
Processes all Excel files in data/input/ directory with comprehensive error handling
"""

import os
import sys
import time
import signal
import threading
import traceback
import psutil
import torch
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

class ProcessMonitor:
    def __init__(self, timeout_minutes=15):
        self.timeout_seconds = timeout_minutes * 60
        self.start_time = time.time()
        self.last_activity = time.time()
        self.is_processing = False
        self.current_file = ""
        self.total_files = 0
        self.processed_files = 0
        self.error_dir = Path("logs/errors")
        self.error_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_process, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info(f"Process monitor started with {timeout_minutes} minute timeout")
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"sema_process_{timestamp}.log"
        
        # Setup logger
        self.logger = logging.getLogger('sema_monitor')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.log_file = log_file
    
    def update_activity(self, message=""):
        """Update last activity timestamp"""
        self.last_activity = time.time()
        if message:
            self.logger.info(f"Activity: {message}")
    
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
            elapsed_total = current_time - self.start_time
            elapsed_since_activity = current_time - self.last_activity
            
            # Log system status
            if elapsed_since_activity > 60:  # Log every minute of inactivity
                self.log_system_status()
            
            # Check for timeout
            if self.is_processing and elapsed_since_activity > self.timeout_seconds:
                self.handle_timeout()
                break
            
            # Check for system resource issues
            self.check_system_resources()
    
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
            
            self.logger.info(status_msg)
            
            if self.current_file:
                progress_msg = f"Current: {self.current_file} ({self.processed_files}/{self.total_files})"
                self.logger.info(progress_msg)
                
        except Exception as e:
            self.logger.error(f"Error logging system status: {e}")
    
    def check_system_resources(self):
        """Check for system resource issues"""
        try:
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                self.logger.warning(f"High memory usage: {memory.percent}%")
            
            # Check GPU memory if available
            if torch.cuda.is_available():
                gpu_memory_percent = (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100
                if gpu_memory_percent > 90:
                    self.logger.warning(f"High GPU memory usage: {gpu_memory_percent:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error checking system resources: {e}")
    
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
1. Restart the process
2. Check GPU memory and restart if needed
3. Process files in smaller batches
4. Check system logs for CUDA errors
"""
        
        self.logger.error(error_msg)
        print(error_msg)
        
        # Save detailed error report
        self.save_error_report("timeout")
        
        # Attempt graceful shutdown
        self.cleanup_and_exit(1)
    
    def save_error_report(self, error_type):
        """Save detailed error report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = self.error_dir / f"error_{error_type}_{timestamp}.json"
        
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
                "log_file": str(self.log_file)
            }
            
            if torch.cuda.is_available():
                error_data["gpu_info"] = {
                    "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                    "memory_cached_gb": torch.cuda.memory_reserved() / 1024**3,
                    "device_name": torch.cuda.get_device_name(0)
                }
            
            with open(error_file, 'w') as f:
                json.dump(error_data, f, indent=2)
            
            self.logger.info(f"Error report saved to: {error_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save error report: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle system signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.cleanup_and_exit(0)
    
    def cleanup_and_exit(self, exit_code):
        """Cleanup and exit"""
        try:
            self.logger.info("Starting cleanup process...")
            
            # Clear GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("GPU cache cleared")
            
            self.logger.info(f"Process completed with exit code: {exit_code}")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
        
        finally:
            os._exit(exit_code)

def safe_import_sema():
    """Safely import SEMA CLI with error handling"""
    try:
        from colab_cli import SemaColabCLI
        return SemaColabCLI
    except ImportError as e:
        print(f"❌ Error importing SEMA CLI: {e}")
        print("Make sure colab_cli.py is in the current directory")
        return None
    except Exception as e:
        print(f"❌ Unexpected error importing SEMA CLI: {e}")
        return None

def process_file_safely(sema, filename, monitor, file_index, total_files):
    """Process a single file with safety checks"""
    monitor.set_current_file(filename, file_index, total_files)
    
    try:
        monitor.start_processing()
        
        # Add heartbeat during processing
        def heartbeat():
            while monitor.is_processing:
                time.sleep(60)  # Update every minute
                if monitor.is_processing:
                    monitor.update_activity(f"Still processing {filename}")
        
        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        
        # Process the file
        success = sema.process_file(filename)
        
        monitor.end_processing()
        monitor.update_activity(f"Completed {filename}: {'SUCCESS' if success else 'FAILED'}")
        
        return success
        
    except Exception as e:
        monitor.end_processing()
        error_msg = f"Error processing {filename}: {str(e)}"
        monitor.logger.error(error_msg)
        monitor.logger.error(traceback.format_exc())
        
        # Save individual file error
        error_file = monitor.error_dir / f"file_error_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(error_file, 'w') as f:
            f.write(f"Error processing {filename}\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}")
        
        return False

def main():
    """Main function with comprehensive safety features"""
    print("🤖 SEMA Safe Batch Processing Started")
    print("=" * 40)
    
    # Initialize process monitor
    monitor = ProcessMonitor(timeout_minutes=15)
    
    try:
        monitor.update_activity("Initializing SEMA CLI")
        
        # Safely import and initialize SEMA
        SemaColabCLI = safe_import_sema()
        if not SemaColabCLI:
            return 1
        
        print("🚀 Initializing SEMA...")
        sema = SemaColabCLI()
        monitor.update_activity("SEMA CLI initialized successfully")
        
        # Check for input files
        input_files = [f for f in os.listdir('data/input') if f.endswith('.xlsx') and not f.startswith('~')]
        
        if not input_files:
            print("❌ No Excel files found in data/input directory")
            print("Please add Excel files to data/input/ and try again")
            monitor.logger.warning("No input files found")
            return 1
        
        print(f"📂 Found {len(input_files)} files to process:")
        for i, file in enumerate(input_files, 1):
            print(f"   {i}. {file}")
        
        monitor.logger.info(f"Found {len(input_files)} input files")
        
        print(f"\n🔄 Processing files with {monitor.timeout_seconds/60} minute timeout per file...")
        print("📊 Progress will be logged and monitored automatically")
        print(f"📁 Logs saved to: {monitor.log_file}")
        print(f"🚨 Error reports saved to: {monitor.error_dir}")
        
        success_count = 0
        failed_files = []
        
        for i, filename in enumerate(input_files, 1):
            print(f"\n📄 Processing {i}/{len(input_files)}: {filename}")
            
            if process_file_safely(sema, filename, monitor, i, len(input_files)):
                success_count += 1
                print(f"✅ {filename} completed successfully")
            else:
                failed_files.append(filename)
                print(f"❌ {filename} failed")
        
        # Final summary
        print(f"\n🎉 Processing Summary:")
        print(f"✅ Successfully processed: {success_count}/{len(input_files)} files")
        
        if failed_files:
            print(f"❌ Failed files: {len(failed_files)}")
            for failed_file in failed_files:
                print(f"   • {failed_file}")
        
        # Show output files
        if success_count > 0:
            output_files = [f for f in os.listdir('data/output') if f.endswith('.xlsx')]
            print(f"📁 Generated {len(output_files)} output files:")
            for file in output_files:
                print(f"   • {file}")
                
            print(f"\n✅ Processing complete!")
            print(f"📂 Output files are in: {os.path.abspath('data/output')}")
            monitor.logger.info(f"Processing completed: {success_count}/{len(input_files)} files successful")
        
        return 0 if success_count > 0 else 1
            
    except Exception as e:
        error_msg = f"Critical error during processing: {e}"
        print(f"❌ {error_msg}")
        monitor.logger.error(error_msg)
        monitor.logger.error(traceback.format_exc())
        monitor.save_error_report("critical")
        return 1
    
    finally:
        monitor.update_activity("Process ending")

if __name__ == "__main__":
    exit_code = main()
    print(f"\n🏁 Process completed with exit code: {exit_code}")
    sys.exit(exit_code)