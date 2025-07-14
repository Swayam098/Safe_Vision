# models/object_detector.py

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch

# Load processor and model once
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.eval()

def detect_objects(image, device="cpu"):
    model.to(device)  # move model to GPU or CPU
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.shape[:2]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    return results
