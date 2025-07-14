import cv2
import torch
from models.object_detector import detect_objects
from models.video_classifier import classify_video_segment  # optional
from utils.log_writer import log_event

def process_video_stream(video_source=0, stop_event=None, device="cpu"):
    cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)

    log_event("System", f"Using device: {device}")
    log_event("System", "Monitoring started.")
    print("🟢 Monitoring started.")

    while cap.isOpened():
        # ✅ Exit loop if stop signal is triggered
        if stop_event and stop_event.is_set():
            break

        ret, frame = cap.read()
        if not ret:
            break

        # ✅ Run object detection on the frame
        results = detect_objects(frame, device=device)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score.item() > 0.85:
                msg = f"{label.item()} detected with confidence {score.item():.2f}"
                print(msg)
                log_event("Detection", msg)

        # 🔄 Optional: run video classification later
        # classify_video_segment(frames)

    cap.release()
    log_event("System", "Monitoring stopped.")
    print("🔴 Monitoring stopped.")
