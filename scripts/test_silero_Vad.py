import torch
import numpy as np
import sounddevice as sd
import time
'''
Captures audio in 100 ms chunks (1600 samples at 16 kHz)

Runs Silero's model(audio_chunk, sample_rate)

Compares speech probability to a threshold (default 0.6)

Logs result in the terminal

'''
# Load the Silero VAD model and utilities
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', 
                              model='silero_vad', 
                              force_reload=False)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils    

# Configuration
SAMPLE_RATE = 16000
WINDOW_SIZE_SAMPLES = 512  # 32ms
CHUNK_SIZE = 16000  # 1 second

# Create audio buffer for live input
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status {status}")
    print(f"Audio input shape:{type(indata)} {indata.shape} and type: {indata.dtype}")
    samples = indata[:, 0]  # Assuming mono input
    chunk = torch.from_numpy(samples.copy()).float()
    if chunk.shape[0] < WINDOW_SIZE_SAMPLES:
        return
    
    speech_prob = model(chunk, SAMPLE_RATE).item()
    print(f"{'🔊 Speech' if speech_prob > 0.6 else '🤫 Silence'} ({speech_prob:.2f})")
    
print("Sarting real time Silero VAD... Speak into the mic")

# for ReSpeaker here is the output: ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:2,0) — 6 channels
devices = sd.query_devices()
mic_index = next(i for i, d in enumerate(devices) if "ReSpeaker" in d["name"])
print(f"Using device: {devices[mic_index]['name']} (index {mic_index})")
with sd.InputStream(
    device=mic_index,
    channels=1,
    samplerate=SAMPLE_RATE,
    blocksize=WINDOW_SIZE_SAMPLES,
    dtype='float32',
    callback=audio_callback
):
    try:
        while True:
            time.sleep(0.1)  # Keep the stream alive
    except KeyboardInterrupt:
        print("Stopping real time Silero VAD.")