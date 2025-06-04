import os
import sys
import time
import threading
import webview
import subprocess
import logging
import signal
import atexit
import requests
from datetime import datetime

# Creating logs directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Generating dynamic log filename
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOG_DIR, f"CreditApprovalAnalysis_{timestamp}.log")

# Logging configuration - Only log to file to avoid stderr interference
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w")
        # Removed StreamHandler to prevent stderr conflicts
    ]
)

# Console logging function
def log_to_console(message, level="INFO"):
    """Print to console and log to file"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {level} - {message}")
    if level == "INFO":
        logging.info(message)
    elif level == "ERROR":
        logging.error(message)
    elif level == "DEBUG":
        logging.debug(message)

# Streamlit port configuration
STREAMLIT_PORT = 8501
STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"

# Global variable to store the Streamlit process
streamlit_process = None

# Path handling for both development and frozen environments
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
    MAIN_SCRIPT = os.path.join(BASE_DIR, "src","app.py")
    STREAMLIT_EXE = os.path.join(BASE_DIR, "streamlit.exe")
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MAIN_SCRIPT = os.path.join(BASE_DIR,"src", "app.py")
    STREAMLIT_EXE = "streamlit"

# Function to terminate Streamlit process
def cleanup_streamlit():
    global streamlit_process
    if streamlit_process:
        log_to_console("Terminating Streamlit process...")
        try:
            # Kill the Streamlit process properly
            if sys.platform == "win32":
                # On Windows, use taskkill to ensure all child processes are terminated
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(streamlit_process.pid)], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
            else:
                # On Unix-like systems
                os.killpg(os.getpgid(streamlit_process.pid), signal.SIGTERM)
            
            streamlit_process = None
            log_to_console("Streamlit process terminated successfully")
        except Exception as e:
            log_to_console(f"Error terminating Streamlit process: {str(e)}", "ERROR")

# Function to check if Streamlit server is ready
def wait_for_streamlit(max_attempts=60, delay=1):
    """Wait for Streamlit server to be ready by checking if it responds to requests"""
    log_to_console("Waiting for Streamlit to fully load...")
    
    for attempt in range(max_attempts):
        try:
            # First check if server responds
            response = requests.get(STREAMLIT_URL, timeout=5)
            if response.status_code == 200:
                # Check if the response contains Streamlit content
                if "streamlit" in response.text.lower() or len(response.text) > 1000:
                    log_to_console("Streamlit server is ready and content loaded!")
                    # Give it a bit more time to fully initialize
                    time.sleep(3)
                    return True
                else:
                    log_to_console(f"Server responding but content not ready yet... (attempt {attempt + 1})")
        except requests.exceptions.RequestException as e:
            log_to_console(f"Connection attempt {attempt + 1} failed: {str(e)}")
        
        time.sleep(delay)
    
    log_to_console("Streamlit server failed to start within the expected time", "ERROR")
    return False

# Function to verify Streamlit app is working
def verify_streamlit_app():
    """Additional verification that the Streamlit app is actually working"""
    try:
        response = requests.get(STREAMLIT_URL, timeout=10)
        log_to_console(f"Final verification - Status: {response.status_code}, Content length: {len(response.text)}")
        
        # Check for common Streamlit elements
        if any(keyword in response.text.lower() for keyword in ['streamlit', 'script', 'react']):
            log_to_console("Streamlit app verification successful!")
            return True
        else:
            log_to_console("Streamlit app verification failed - no expected content found")
            return False
    except Exception as e:
        log_to_console(f"Verification failed: {str(e)}", "ERROR")
        return False

# Function to run Streamlit in background
def start_streamlit_background():
    global streamlit_process
    try:
        log_to_console(f"Starting Streamlit server with script: {MAIN_SCRIPT}")
        
        if getattr(sys, 'frozen', False):
            cmd = [STREAMLIT_EXE, "run", MAIN_SCRIPT, 
                  "--server.port", str(STREAMLIT_PORT), 
                  "--server.headless", "true",
                  "--server.runOnSave", "false",
                  "--browser.gatherUsageStats", "false"]
        else:
            cmd = ["streamlit", "run", MAIN_SCRIPT, 
                  "--server.port", str(STREAMLIT_PORT), 
                  "--server.headless", "true",
                  "--server.runOnSave", "false",
                  "--browser.gatherUsageStats", "false"]
        
        # Check if file exists
        if not os.path.exists(MAIN_SCRIPT):
            log_to_console(f"Main script not found at: {MAIN_SCRIPT}", "ERROR")
            return
            
        log_to_console(f"Running command: {' '.join(cmd)}")
        
        # Start the process with minimal output capture
        if sys.platform == "win32":
            # For Windows - redirect output to DEVNULL to avoid conflicts
            streamlit_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            # For Unix-like systems
            streamlit_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )
        
        log_to_console("Streamlit process started successfully")
            
    except Exception as e:
        log_to_console(f"Error starting Streamlit: {str(e)}", "ERROR")

# Register cleanup function for normal exit
atexit.register(cleanup_streamlit)

# Define window close event handler
def on_closed():
    log_to_console("Window closed, cleaning up...")
    cleanup_streamlit()
    log_to_console("Application shutdown complete")

def main():
    """Main function - ensures PyWebView runs on main thread"""
    
    # Verify we're on the main thread
    if threading.current_thread() != threading.main_thread():
        log_to_console("ERROR: main() is not running on the main thread!", "ERROR")
        return
    
    log_to_console("Starting application on main thread...")
    
    try:
        # Start Streamlit in a background thread
        log_to_console("Starting Streamlit in background thread...")
        streamlit_thread = threading.Thread(target=start_streamlit_background, daemon=True)
        streamlit_thread.start()
        
        # Wait a moment for the process to start
        time.sleep(5)  # Increased wait time
        
        # Wait for Streamlit server to be ready
        log_to_console("Checking if Streamlit server is ready...")
        if not wait_for_streamlit():
            log_to_console("Streamlit server is not responding", "ERROR")
            cleanup_streamlit()
            return
        
        # Additional verification
        if not verify_streamlit_app():
            log_to_console("Streamlit app verification failed", "ERROR")
            cleanup_streamlit()
            return
        
        # Verify we're still on main thread before starting PyWebView
        if threading.current_thread() != threading.main_thread():
            log_to_console("ERROR: Not on main thread before PyWebView start!", "ERROR")
            return
            
        # Start PyWebView on the main thread
        log_to_console(f"Starting PyWebView with URL {STREAMLIT_URL}")
        log_to_console(f"Current thread: {threading.current_thread().name}")
        log_to_console(f"Is main thread: {threading.current_thread() == threading.main_thread()}")
        
        # Test URL one more time before starting PyWebView
        try:
            test_response = requests.get(STREAMLIT_URL, timeout=5)
            log_to_console(f"Final URL test before PyWebView - Status: {test_response.status_code}")
        except Exception as e:
            log_to_console(f"Final URL test failed: {str(e)}", "ERROR")
        
        window = webview.create_window(
            "Credit Approval Analysis", 
            STREAMLIT_URL,
            width=1200,
            height=800,
            min_size=(800, 600)
        )
        window.events.closed += on_closed
        
        # This MUST run on the main thread
        webview.start(debug=False)  # Added debug flag for more info
        
    except Exception as e:
        log_to_console(f"Error in main function: {str(e)}", "ERROR")
        import traceback
        log_to_console(f"Full traceback: {traceback.format_exc()}", "ERROR")
    finally:
        # Final cleanup
        cleanup_streamlit()

if __name__ == "__main__":
    # Ensure this script is called directly and not imported
    log_to_console("Script starting...")
    log_to_console(f"Python version: {sys.version}")
    log_to_console(f"Current thread at startup: {threading.current_thread().name}")
    log_to_console(f"Is main thread: {threading.current_thread() == threading.main_thread()}")
    
    main()