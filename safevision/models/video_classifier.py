from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import torch
import cv2

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base")

def classify_video_segment(frames):
    inputs = processor(frames, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]
