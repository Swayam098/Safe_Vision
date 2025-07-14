import datetime
import streamlit as st
import threading
from utils.log_writer import log_event

# Function to write alerts to Streamlit UI and logs
def trigger_alert(event_type: str, description: str, streamlit_log=True):
    # Log the event using centralized logger
    log_event(event_type, description)

    # Optionally show toast in Streamlit UI
    if streamlit_log:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        threading.Thread(
            target=st.toast,
            args=(f"🚨 [{timestamp}] {event_type}: {description}",),
            daemon=True
        ).start()
