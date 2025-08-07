import sounddevice as sd
import numpy as np
import queue

def stream_microphone_chunks(
    chunk_duration: float = 1.0,
    sample_rate: int = 16000,
    channels: int = 1,
    device: int = None
):
    """
    Stream audio from the microphone and yield fixed-length chunks in real time.

    Args:
        chunk_duration (float): Duration of each chunk in seconds.
        sample_rate (int): Sampling rate of the audio.
        channels (int): Number of channels (1 = mono).
        device (int): Optional device index for the microphone.

    Yields:
        np.ndarray: Audio chunk with shape (chunk_samples, channels).
    """
    chunk_samples = int(chunk_duration * sample_rate)
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"⚠️ {status}")
        audio_queue.put(indata.copy())

    with sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        callback=audio_callback,
        blocksize=chunk_samples,
        dtype='float32',
        device=device
    ):
        print("🎙 Streaming microphone audio... Press Ctrl+C to stop.")
        try:
            while True:
                chunk = audio_queue.get()
                yield chunk
        except KeyboardInterrupt:
            print("🛑 Stopped streaming.")


# Example usage:
# for i, chunk in enumerate(stream_microphone_chunks(chunk_duration=1.0)):
#     print(f"Chunk {i} shape: {chunk.shape}")
#     if i == 5:
#         break
