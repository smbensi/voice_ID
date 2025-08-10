import sounddevice as sd
import numpy as np
from queue import Queue
import sys
import os
import json
import pandas as pd
import librosa
import soundfile as sf
import time 

from speaker_verification.mqtt.mqtt_handler import init_mqtt_connection
from speaker_verification.settings import mqtt_settings, recognition_params

from speaker_verification import LOGGER
from speaker_verification.pipelines.voice_recognition import VoiceRecognition
# A thread-safe queue to hold the audio data
audio_queue = Queue()
start = time.time()

def audio_callback(indata, frames, time, status):
    """
    This function is called by the sounddevice library for each new block of audio.
    It adds the audio data (a numpy array) to a queue for processing.
    """
    if status:
        print(status, file=sys.stderr)

    # Put the audio data into the queue
    audio_queue.put(indata.copy())
    
class ProcessStream:
    def __init__(self, speaker_verification_pipe):
        self.audio_data = np.empty((0,), dtype=np.float32) 
        self.speaker_verification_pipe = speaker_verification_pipe
        self.index = 0
    def process_audio_stream(self,audio_data):
        if len(self.audio_data) < 5*16000:
            audio_data = np.squeeze(audio_data)
            self.audio_data = np.concatenate((self.audio_data, audio_data), axis=0)
        else:
            result = voice_verification_pipeline.is_recognized(self.audio_data)
            audio_file = f"output_{self.index}.wav"
            print(f"Verification result for microphone {audio_file}: {result}")
            # sf.write(f'/code/audio_streaming_data/{audio_file}', self.audio_data, samplerate=16000)
            self.index += 1
            self.audio_data = np.empty((0,), dtype=np.float32) 
            


def start_audio_stream(mic_processor, samplerate=16000, channels=1, blocksize=1024):
    """
    Starts the audio stream from the microphone and processes it in real-time.
    """
    try:
        print("Starting audio stream... Press Ctrl+C to stop.")

        # Use sounddevice.InputStream to capture audio
        with sd.InputStream(device="ReSpeaker",
                            samplerate=samplerate,
                            channels=channels,
                            blocksize=blocksize,
                            callback=audio_callback):
            LOGGER.debug(f"time of loading = {time.time() - start:.1f} sec")
            # Continuously get audio data from the queue and process it
            while True:
                audio_data = audio_queue.get()
                mic_processor.process_audio_stream(audio_data)

    except KeyboardInterrupt:
        print("\nStopping audio stream.")
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

def process_audio_file(audio_file, voice_verification_pipeline):
    """
    Process an audio file for speaker verification.
    """
    duration = librosa.get_duration(filename=audio_file)
    # Pass the audio data to the voice verification pipeline
    if recognition_params.RECOGNIZE:
        result = voice_verification_pipeline.is_recognized(audio_file)
        print(f"Verification result for {audio_file} {duration=}: {result}")
        if isinstance(result, str):
            mqtt_settings.VOICE_CLIENT.publish(mqtt_settings.TOPICS_TO_BRAIN["VOICE_RECOGNIZED"],result)

def process_registration(name, audio_file, voice_verification):
    embedding_file = recognition_params.EMBEDDING_FILE
    if not os.path.exists(embedding_file):
        df = pd.DataFrame(columns=["name", "embedding"])
    else:
        df = pd.read_json("embeddings.json")
        # df.to_json(embedding_file, orient="records", indent=2)

    # If you want to register a new speaker, you can call the create_embedding method
    embedding = voice_verification_pipeline.get_embedding_from_audio_file(audio_file)
    print(f"Created embedding for {audio_file}: {type(embedding)}")
    print(df)
    df.loc[len(df)] = [name, embedding.tolist()]
    df.to_json(embedding_file, orient="records", indent=2)

if __name__ == "__main__":
    
    mqtt_settings.VOICE_CLIENT = init_mqtt_connection(name="voice_verif")
    voice_verification_pipeline = VoiceRecognition(model_name="titanet_small")
    audio_source = os.getenv("SOURCE","file")  # Change to "file" if you want to read from a file and "mic" for microphone input
    audio_path = "/home/mat/Documents/voice_ID/data/long_audio/adina_sagi/2"
    audio_path = os.getenv("AUDIO_PATH","/code/jake")
    registration  = False
    if registration:
        name = "jake"
        process_registration(name, audio_path, voice_verification_pipeline)

    elif audio_source == "file":
        if os.path.isdir(audio_path):
            for filename in os.listdir(audio_path):
                filename = os.path.join(audio_path,filename)
                process_audio_file(filename, voice_verification_pipeline)
        else:
            process_audio_file(audio_path, voice_verification_pipeline)
            
    elif audio_source == "mic":
        # You can customize these parameters based on your needs
        mic_processor = ProcessStream(voice_verification_pipeline)
        SAMPLERATE = 16000  # Common for speech recognition
        CHANNELS = 1        # Mono audio
        BLOCKSIZE = 1024    # Number of frames per block
        start_audio_stream(mic_processor, samplerate=SAMPLERATE, channels=CHANNELS, blocksize=BLOCKSIZE)