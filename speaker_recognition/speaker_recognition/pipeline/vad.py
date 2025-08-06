import torch
import torchaudio
import numpy as np
import sounddevice as sd
import time
import soundfile as sf


class SileroVAD:
    '''
    Captures audio in 100 ms chunks (1600 samples at 16 kHz)
    Runs Silero's model(audio_chunk, sample_rate)
    Compares speech probability to a threshold (default 0.6)
    Logs result in the terminal
    '''
    
    def __init__(self):
        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', 
                                                model='silero_vad', 
                                                force_reload=False)
        self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks = self.utils
        self.sample_rate = 16000
        self.window_size_samples = 512  # 32ms
        self.chunk_size = 16000  # 1 second
    
    
    def strip_silence(self, waveform: torch.Tensor) -> bool:
        # waveform, sample_rate = torchaudio.load(audio_path)  # or .mp3, .aac, etc.
        print("waveform shape:", waveform.shape, "and type:", waveform.dtype)
        samples = waveform[:]  # Assuming mono input
        chunk = torch.from_numpy(samples.copy()).float()
        if chunk.shape[0] < self.window_size_samples:
            return

        speech_prob = self.model(chunk, self.sample_rate).item()
        print(f"{'🔊 Speech' if speech_prob > 0.6 else '🤫 Silence'} ({speech_prob:.2f})")
        if speech_prob > 0.6:
            return True
        return False
        
if __name__ == "__main__":
    audio_file = "/home/mat/Documents/voice_ID/data/mathias/mathias_respeaker_sample.wav"
    # audio_file = "/home/mat/Documents/voice_ID/data/trump/trump2_mono.mp3"
    waveform, sample_rate = torchaudio.load(audio_file)
    print(f"Loaded audio file: {audio_file} with sample rate: {sample_rate} Hz"
          f" and waveform shape: {waveform.shape} and type: {waveform.dtype}")
    vad = SileroVAD()
    
    cleaned_audio = np.empty(0)
    created = False
    speech = 0
    silence = 0
    for i, chunk in enumerate(waveform.split(int(0.032*16000), dim=1)):  # Split into 1-second chunks
        if not vad.strip_silence(chunk.numpy().squeeze()) :
            print(f"Chunk {i + 1} is silence, skipping...")
            silence += 1
        else:
            cleaned_audio = np.concatenate((cleaned_audio, chunk.numpy().squeeze()))
            print(f"Chunk {i + 1} is speech, processing...")
            created = True
            speech += 1

    print(f"Processed {i + 1} chunks: {speech} speech and {silence} silence")

    sf.write("cleaned_output2.wav", cleaned_audio, sample_rate)