import torchaudio
from transformers import Wav2Vec2ForAudioFrameClassification, Wav2Vec2Processor
import torch

processor = Wav2Vec2Processor.from_pretrained("MIT/ast-finetuned-audioset-10-10")
model = Wav2Vec2ForAudioFrameClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10")

def detect_anomaly(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.sigmoid(logits).mean(dim=1).squeeze()
    return pred  # Returns probability per label
