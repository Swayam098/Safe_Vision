import datetime
import os

LOG_FILE = "logs.txt"

def log_event(event_type, description):
    ensure_log_exists()
    timestamp = datetime.datetime.now().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {event_type}: {description}\n")

def read_logs():
    ensure_log_exists()
    with open(LOG_FILE, "r") as f:
        return f.read()

def ensure_log_exists():
    if not os.path.exists(LOG_FILE):
        open(LOG_FILE, "w").close()
