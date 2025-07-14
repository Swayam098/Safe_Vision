import streamlit as st
from detector import process_video_stream
from utils.log_writer import read_logs, log_event
import threading
import torch

# 🚀 Set page config — must be FIRST Streamlit command
st.set_page_config(page_title="SafeVision - AI Safety Monitor", layout="wide")

# 🔁 Setup session state
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
    st.session_state.stop_event = threading.Event()
    st.session_state.thread = None

# 🛡️ App Title
st.title("🛡️ SafeVision – AI Workplace Safety Monitoring")

# ⚙️ Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
log_event("System", f"Using device: {device}")
st.info(f"💻 Using device: `{device.upper()}`")

# ▶️ Start Monitoring
if not st.session_state.monitoring and st.button("🚀 Start Monitoring"):
    st.session_state.stop_event.clear()
    st.session_state.thread = threading.Thread(
        target=process_video_stream,
        args=(0, st.session_state.stop_event, device),
        daemon=True
    )
    st.session_state.thread.start()
    st.session_state.monitoring = True
    log_event("System", "Monitoring started.")
    st.success("🟢 Monitoring started.")

# ⏹️ Stop Monitoring
if st.session_state.monitoring and st.button("🛑 Stop Monitoring"):
    st.session_state.stop_event.set()
    st.session_state.monitoring = False
    log_event("System", "Monitoring stopped.")
    st.warning("🔴 Monitoring stopped.")

# 🔔 Real-time Logs
with st.expander("📄 Real-time Logs"):
    logs = read_logs()
    st.code(logs if logs else "No logs yet.", language="text")

# ✅ Footer
st.markdown("---")
st.markdown(f"✅ **System Status:** All models loaded and running on `{device.upper()}`.")
