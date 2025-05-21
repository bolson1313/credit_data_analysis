import subprocess
import threading
import webview
import time
import os
import sys
import requests
import signal

# Zabezpieczenie przed rekurencyjnym uruchomieniem przez Streamlit
if os.environ.get("STREAMLIT_RUN") == "true":
    sys.exit()
os.environ["STREAMLIT_RUN"] = "true"

# Ścieżka do pliku Streamlit
STREAMLIT_APP_PATH = os.path.join(os.path.dirname(__file__), "app.py")

# Globalny uchwyt procesu Streamlit
streamlit_process = None

def start_streamlit():
    global streamlit_process
    os.environ["BROWSER"] = "none"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

    streamlit_process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", STREAMLIT_APP_PATH,
            "--server.runOnSave", "false"
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def wait_for_streamlit(url="http://localhost:8501", timeout=15):
    for _ in range(timeout * 10):
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.1)
    return False

if __name__ == '__main__':
    threading.Thread(target=start_streamlit, daemon=True).start()

    if wait_for_streamlit():
        try:
            webview.create_window("Analiza danych kredytowych", "http://localhost:8501", width=1024, height=768)
            webview.start()
        finally:
            # Zakończenie procesu Streamlit po zamknięciu okna
            if streamlit_process and streamlit_process.poll() is None:
                streamlit_process.terminate()  # wysyła SIGTERM
                try:
                    streamlit_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    streamlit_process.kill()  # wymuszenie zakończenia
    else:
        print("Streamlit nie uruchomił się poprawnie.")
