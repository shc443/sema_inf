#!/usr/bin/env python3
"""
SEMA Easy Run - Double-Click to Process VOC Files
=================================================

This script provides a simple GUI interface for SEMA VOC analysis.
Just double-click this file to start processing your Excel files!

Features:
- File selection dialog
- Real-time progress display
- Automatic error handling
- Results saved to output folder
- No command line needed!
"""

import os
import sys
import time
import threading
import traceback
from pathlib import Path
from datetime import datetime
import json

# Check if tkinter is available
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("GUI not available, falling back to console mode")

class SEMAEasyGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎯 SEMA VOC Analysis - Easy Run")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Variables
        self.selected_files = []
        self.is_processing = False
        self.output_dir = Path("data/output")
        self.log_dir = Path("logs")
        
        # Setup directories
        self.setup_directories()
        
        # Create GUI
        self.create_gui()
        
        # Center window
        self.center_window()
    
    def setup_directories(self):
        """Create necessary directories"""
        Path("data/input").mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        Path("logs/errors").mkdir(parents=True, exist_ok=True)
    
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_gui(self):
        """Create the main GUI interface"""
        # Title
        title_frame = tk.Frame(self.root)
        title_frame.pack(pady=10)
        
        title_label = tk.Label(title_frame, text="🎯 SEMA VOC Analysis", 
                              font=('Arial', 18, 'bold'))
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="Korean Voice of Customer Sentiment Analysis", 
                                 font=('Arial', 10))
        subtitle_label.pack()
        
        # Instructions
        instructions = tk.Label(self.root, 
                               text="Select your Excel files with VOC1 and VOC2 columns, then click Process!",
                               font=('Arial', 11), fg='blue')
        instructions.pack(pady=5)
        
        # File selection frame
        file_frame = tk.Frame(self.root)
        file_frame.pack(pady=10, padx=20, fill='x')
        
        self.select_button = tk.Button(file_frame, text="📁 Select Excel Files", 
                                      command=self.select_files, font=('Arial', 12),
                                      bg='lightblue', width=20)
        self.select_button.pack(side='left', padx=5)
        
        self.clear_button = tk.Button(file_frame, text="🗑️ Clear Selection", 
                                     command=self.clear_files, font=('Arial', 12),
                                     bg='lightcoral', width=15)
        self.clear_button.pack(side='left', padx=5)
        
        # Selected files display
        files_label = tk.Label(self.root, text="Selected Files:", font=('Arial', 11, 'bold'))
        files_label.pack(anchor='w', padx=20, pady=(10,0))
        
        self.files_listbox = tk.Listbox(self.root, height=6, font=('Arial', 10))
        self.files_listbox.pack(pady=5, padx=20, fill='x')
        
        # Process button
        self.process_button = tk.Button(self.root, text="🚀 Process All Files", 
                                       command=self.start_processing, font=('Arial', 14, 'bold'),
                                       bg='lightgreen', height=2, width=25)
        self.process_button.pack(pady=15)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.pack(pady=5)
        
        self.progress_label = tk.Label(self.root, text="Ready to process files", 
                                      font=('Arial', 10))
        self.progress_label.pack()
        
        # Log display
        log_label = tk.Label(self.root, text="Processing Log:", font=('Arial', 11, 'bold'))
        log_label.pack(anchor='w', padx=20, pady=(15,0))
        
        self.log_text = scrolledtext.ScrolledText(self.root, height=12, font=('Consolas', 9))
        self.log_text.pack(pady=5, padx=20, fill='both', expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             relief='sunken', anchor='w', font=('Arial', 9))
        status_bar.pack(side='bottom', fill='x')
        
        # Results button (initially hidden)
        self.results_button = tk.Button(self.root, text="📂 Open Results Folder", 
                                       command=self.open_results_folder, font=('Arial', 12),
                                       bg='gold', width=20)
    
    def log(self, message):
        """Add message to log display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, formatted_message)
        self.log_text.see(tk.END)
        self.root.update()
        
        # Also save to file
        log_file = self.log_dir / f"sema_gui_{datetime.now().strftime('%Y%m%d')}.log"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(formatted_message)
    
    def select_files(self):
        """Open file dialog to select Excel files"""
        files = filedialog.askopenfilenames(
            title="Select Excel Files for VOC Analysis",
            filetypes=[
                ("Excel files", "*.xlsx"),
                ("Excel files", "*.xls"),
                ("All files", "*.*")
            ]
        )
        
        if files:
            self.selected_files = list(files)
            self.update_files_display()
            self.log(f"Selected {len(files)} files for processing")
    
    def clear_files(self):
        """Clear selected files"""
        self.selected_files = []
        self.update_files_display()
        self.log("File selection cleared")
    
    def update_files_display(self):
        """Update the files listbox"""
        self.files_listbox.delete(0, tk.END)
        for file_path in self.selected_files:
            filename = os.path.basename(file_path)
            size = os.path.getsize(file_path) / 1024  # KB
            self.files_listbox.insert(tk.END, f"{filename} ({size:.1f} KB)")
    
    def start_processing(self):
        """Start processing files in a separate thread"""
        if not self.selected_files:
            messagebox.showwarning("No Files", "Please select Excel files first!")
            return
        
        if self.is_processing:
            messagebox.showinfo("Already Processing", "Processing is already in progress!")
            return
        
        # Disable buttons
        self.process_button.config(state='disabled', text="🔄 Processing...")
        self.select_button.config(state='disabled')
        self.clear_button.config(state='disabled')
        
        # Start processing thread
        self.is_processing = True
        thread = threading.Thread(target=self.process_files, daemon=True)
        thread.start()
    
    def process_files(self):
        """Process all selected files"""
        try:
            self.log("🚀 Starting SEMA VOC Analysis...")
            self.status_var.set("Initializing...")
            
            # Copy files to input directory
            input_dir = Path("data/input")
            for i, file_path in enumerate(self.selected_files):
                filename = os.path.basename(file_path)
                destination = input_dir / filename
                
                self.log(f"📁 Copying {filename} to input directory...")
                
                # Copy file
                import shutil
                shutil.copy2(file_path, destination)
                
                # Update progress
                progress = (i + 1) / len(self.selected_files) * 30  # 30% for copying
                self.progress_var.set(progress)
                self.progress_label.config(text=f"Copying files... {i+1}/{len(self.selected_files)}")
                self.root.update()
            
            self.log("✅ All files copied to input directory")
            
            # Import and initialize SEMA
            self.log("🔧 Initializing SEMA processing engine...")
            self.status_var.set("Loading AI model...")
            self.progress_var.set(40)
            self.progress_label.config(text="Loading AI model...")
            self.root.update()
            
            try:
                from process_batch_safe import ProcessMonitor
                from colab_cli import SemaColabCLI
                
                # Initialize with GUI monitoring
                monitor = ProcessMonitor(timeout_minutes=15)
                sema = SemaColabCLI()
                
                self.log("✅ SEMA engine loaded successfully")
                
            except Exception as e:
                self.log(f"❌ Error loading SEMA: {str(e)}")
                raise
            
            # Process files
            self.log("🔄 Starting file processing...")
            self.status_var.set("Processing files...")
            
            input_files = [f for f in os.listdir('data/input') if f.endswith('.xlsx') and not f.startswith('~')]
            success_count = 0
            
            for i, filename in enumerate(input_files):
                self.log(f"📄 Processing file {i+1}/{len(input_files)}: {filename}")
                
                # Update progress
                base_progress = 50 + (i / len(input_files)) * 45  # 45% for processing
                self.progress_var.set(base_progress)
                self.progress_label.config(text=f"Processing {filename}...")
                self.root.update()
                
                try:
                    if sema.process_file(filename):
                        success_count += 1
                        self.log(f"✅ {filename} processed successfully")
                    else:
                        self.log(f"❌ Failed to process {filename}")
                
                except Exception as e:
                    self.log(f"❌ Error processing {filename}: {str(e)}")
            
            # Complete
            self.progress_var.set(100)
            self.progress_label.config(text="Processing complete!")
            self.status_var.set("Complete")
            
            # Show results
            output_files = list(self.output_dir.glob("*.xlsx"))
            
            if success_count > 0:
                self.log(f"🎉 Processing completed successfully!")
                self.log(f"✅ Processed {success_count}/{len(input_files)} files")
                self.log(f"📁 Generated {len(output_files)} output files:")
                
                for output_file in output_files:
                    size = output_file.stat().st_size / 1024  # KB
                    self.log(f"   • {output_file.name} ({size:.1f} KB)")
                
                self.log(f"📂 Results saved to: {self.output_dir.absolute()}")
                
                # Show results button
                self.results_button.pack(pady=10)
                
                # Success message
                messagebox.showinfo("Success!", 
                                   f"Processing complete!\n\n"
                                   f"✅ Processed: {success_count}/{len(input_files)} files\n"
                                   f"📁 Output files: {len(output_files)}\n"
                                   f"📂 Location: {self.output_dir.absolute()}")
            else:
                self.log("❌ No files were processed successfully")
                messagebox.showerror("Processing Failed", 
                                    "No files were processed successfully.\n\n"
                                    "Please check:\n"
                                    "• Files have VOC1 and VOC2 columns\n"
                                    "• Files contain Korean text\n"
                                    "• Check the log for details")
        
        except Exception as e:
            self.log(f"🚨 Critical error: {str(e)}")
            self.log(f"Traceback: {traceback.format_exc()}")
            
            # Save error report
            error_file = Path("logs/errors") / f"gui_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(error_file, 'w') as f:
                f.write(f"GUI Processing Error\n")
                f.write(f"Time: {datetime.now()}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Traceback:\n{traceback.format_exc()}")
            
            messagebox.showerror("Error", 
                                f"An error occurred during processing:\n\n{str(e)}\n\n"
                                f"Error details saved to:\n{error_file}")
        
        finally:
            # Re-enable buttons
            self.is_processing = False
            self.process_button.config(state='normal', text="🚀 Process All Files")
            self.select_button.config(state='normal')
            self.clear_button.config(state='normal')
    
    def open_results_folder(self):
        """Open the results folder in file explorer"""
        try:
            import subprocess
            import platform
            
            if platform.system() == 'Windows':
                subprocess.run(['explorer', str(self.output_dir.absolute())])
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', str(self.output_dir.absolute())])
            else:  # Linux
                subprocess.run(['xdg-open', str(self.output_dir.absolute())])
                
        except Exception as e:
            messagebox.showinfo("Results Location", 
                               f"Results are saved in:\n{self.output_dir.absolute()}")
    
    def run(self):
        """Start the GUI application"""
        try:
            self.log("🎯 SEMA Easy Run started")
            self.log("Select your Excel files and click Process to begin!")
            self.root.mainloop()
        except KeyboardInterrupt:
            self.log("Application closed by user")

class SEMAConsoleMode:
    """Fallback console mode when GUI is not available"""
    
    def __init__(self):
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories"""
        Path("data/input").mkdir(parents=True, exist_ok=True)
        Path("data/output").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
    
    def run(self):
        """Run in console mode"""
        print("🎯 SEMA VOC Analysis - Console Mode")
        print("=" * 40)
        print()
        print("GUI not available, running in console mode.")
        print()
        print("📁 Please copy your Excel files to the 'data/input' folder")
        print("   Example: copy your .xlsx files to data/input/")
        print()
        
        input("Press Enter when you've copied your files...")
        
        # Check for files
        input_files = [f for f in os.listdir('data/input') if f.endswith('.xlsx')]
        
        if not input_files:
            print("❌ No Excel files found in data/input/")
            input("Press Enter to exit...")
            return
        
        print(f"📂 Found {len(input_files)} files:")
        for i, filename in enumerate(input_files, 1):
            print(f"   {i}. {filename}")
        
        print()
        confirm = input("Process these files? (y/n): ")
        
        if confirm.lower() != 'y':
            print("Processing cancelled.")
            return
        
        # Process files
        try:
            from process_batch_safe import main as process_main
            
            print("\n🚀 Starting processing...")
            result = process_main()
            
            if result == 0:
                print("\n✅ Processing completed successfully!")
                print("📂 Check the data/output/ folder for results")
            else:
                print("\n❌ Processing failed. Check logs for details.")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        input("\nPress Enter to exit...")

def main():
    """Main entry point"""
    print("🎯 SEMA VOC Analysis - Easy Run")
    print("Starting application...")
    
    if GUI_AVAILABLE:
        try:
            app = SEMAEasyGUI()
            app.run()
        except Exception as e:
            print(f"GUI failed: {e}")
            print("Falling back to console mode...")
            app = SEMAConsoleMode()
            app.run()
    else:
        app = SEMAConsoleMode()
        app.run()

if __name__ == "__main__":
    main()