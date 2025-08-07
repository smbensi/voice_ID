import sounddevice as sd
import numpy as np
from queue import Queue
import sys

# A thread-safe queue to hold the audio data
audio_queue = Queue()

def audio_callback(indata, frames, time, status):
    """
    This function is called by the sounddevice library for each new block of audio.
    It adds the audio data (a numpy array) to a queue for processing.
    """
    if status:
        print(status, file=sys.stderr)

    # Put the audio data into the queue
    audio_queue.put(indata.copy())

def process_audio_stream(audio_data):
    """
    This is your custom function to process the streaming audio.
    You can replace this with your own logic (e.g., speech recognition, etc.).
    """
    # For this example, we'll just print the shape and type of the audio data.
    # The audio_data is a numpy array.
    print(f"Received audio block with shape: {audio_data.shape} and data type: {audio_data.dtype}")
    # Example: You could send this data to a speech recognition model.
    # recognition_result = my_asr_model(audio_data)
    # print(recognition_result)

def start_audio_stream(samplerate=16000, channels=1, blocksize=1024):
    """
    Starts the audio stream from the microphone and processes it in real-time.
    """
    try:
        print("Starting audio stream... Press Ctrl+C to stop.")

        # Use sounddevice.InputStream to capture audio
        with sd.InputStream(samplerate=samplerate,
                            channels=channels,
                            blocksize=blocksize,
                            callback=audio_callback):

            # Continuously get audio data from the queue and process it
            while True:
                audio_data = audio_queue.get()
                process_audio_stream(audio_data)

    except KeyboardInterrupt:
        print("\nStopping audio stream.")
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    # You can customize these parameters based on your needs
    SAMPLERATE = 16000  # Common for speech recognition
    CHANNELS = 1        # Mono audio
    BLOCKSIZE = 1024    # Number of frames per block

    start_audio_stream(samplerate=SAMPLERATE, channels=CHANNELS, blocksize=BLOCKSIZE)