from pydub import AudioSegment
import numpy as np
import sounddevice as sd
import soundfile as sf
from typing import Optional

def convert_audio_to_mono_16khz(
    source: str = "file",
    file_path: Optional[str] = None,
    mic_duration: float = 5.0,
    sample_rate: int = 16000,
    channels: int = 1,
) -> np.ndarray:
    """
    Load audio from file or microphone and convert to mono with 16kHz sample rate.

    Args:
        source (str): "file" or "mic"
        file_path (str): Path to input file if source="file"
        mic_duration (float): Duration to record from microphone in seconds if source="mic"
        sample_rate (int): Target sample rate (default 16000)
        channels (int): Target number of channels (default 1 for mono)

    Returns:
        np.ndarray: Numpy array of shape (samples, 1) with float32 values in range [-1.0, 1.0]
    """
    if source == "file":
        if not file_path:
            raise ValueError("file_path must be provided when source='file'")
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(channels).set_frame_rate(sample_rate)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)
        return samples.reshape(-1, 1)

    elif source == "mic":
        print(f"🎙 Recording from mic: {mic_duration} seconds at {sample_rate} Hz...")
        recording = sd.rec(int(mic_duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
        sd.wait()
        return recording.reshape(-1, 1)

    else:
        raise ValueError("Invalid source. Must be 'file' or 'mic'.")


# Example usage:
# file_audio = convert_audio_to_mono_16khz(source="file", file_path="example.wav")
# mic_audio = convert_audio_to_mono_16khz(source="mic", mic_duration=3.0)
