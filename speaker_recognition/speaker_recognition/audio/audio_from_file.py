import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import io

def get_audio_chunks(source="mic", file_path=None, chunk_duration=5, sample_rate=16000, channels=1):
    """
    Records audio from a microphone or loads from a file, then splits it into chunks.
    
    Parameters:
        source (str): "mic" or "file"
        file_path (str): path to the audio file if source="file"
        chunk_duration (float): duration of each chunk in seconds
        sample_rate (int): audio sampling rate
        channels (int): number of channels (1 = mono, 2 = stereo)
    
    Returns:
        List of numpy arrays, each containing one chunk of audio
    """
    audio_data = None
    
    # 1. Record from microphone
    if source == "mic":
        duration = float(input(f"Enter recording duration in seconds: "))
        print(f"🎙 Recording {duration} seconds...")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
        sd.wait()
    
    # 2. Load from audio file
    elif source == "file":
        if not file_path:
            raise ValueError("file_path must be provided when source='file'")
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(sample_rate).set_channels(channels)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
        audio_data = samples.reshape((-1, channels))
    else:
        raise ValueError("source must be 'mic' or 'file'")
    
    # Ensure we have a numpy array
    if not isinstance(audio_data, np.ndarray):
        raise RuntimeError("Audio data not loaded correctly")

    # 3. Split into chunks
    chunk_samples = int(chunk_duration * sample_rate)
    chunks = [
        np.squeeze(audio_data[i:i + chunk_samples])
        for i in range(0, len(audio_data), chunk_samples)
        if len(audio_data[i:i + chunk_samples]) == chunk_samples
    ]
    
    print(f"✅ Created {len(chunks)} chunks of {chunk_duration} seconds each.")
    return chunks

if __name__ == "__main__":
    # Example usage:
    audio_file = "/home/mat/Documents/voice_ID/trump1.mp3"
    chunks = get_audio_chunks(source="file", file_path=audio_file, chunk_duration=5)
    # chunks = get_audio_chunks(source="mic", chunk_duration=3)
