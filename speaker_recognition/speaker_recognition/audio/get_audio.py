import alsaaudio
import errno
import time
import threading
import numpy as np
import wave
from audio_buffer import AudioBuffer

# from speaker_recognition.utils import LOGGER as logger
import logging
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
class AudioConfig:
    def __init__(self, sample_rate=16000, channels=6, period_size=1024, channel_index=0, device_name='default'):
        self.sample_rate = sample_rate
        self.channels = channels
        self.period_size = period_size
        self.device_name = device_name
        self.channel_index = channel_index


class AudioManager(threading.Thread):
    def __init__(self, audio_config=None, audio_buffer=None):
        super().__init__(daemon=True)
        self.audio_config = audio_config or AudioConfig()
        self.buffer = audio_buffer or AudioBuffer()
        self.should_listen = True
        self.running = False
        self.pcm = self.try_open_initial_device()
    
    def try_open_initial_device(self):
        logger.info("Trying initial mic connection...")
        pcm = self.open_device()
        if pcm is None:
            logger.critical("Initial mic connection failed. Raising error.")
            raise RuntimeError("Mic device failed to connect on first attempt.")
        logger.info("Initial mic connection succeeded.")
        return pcm

    def run(self):
        self.running = True
        while self.running:
            if self.pcm is None:
                logger.warning("Attempting to open mic device...")
                self.pcm = self.open_device()
                if self.pcm:
                    logger.info("Mic device successfully opened.")
                else:
                    logger.error("Mic device still unavailable. Retrying in 5 seconds...")
                    time.sleep(5)
                    continue

            try:
                length, data = self.pcm.read()

                if length > 0:
                    if not self.should_listen:
                        logger.info("Dropping chunk (should_listen is False)")
                        continue

                    samples = np.frombuffer(data, dtype=np.int16)

                    try:
                        frames = samples.reshape(-1, self.audio_config.channels)
                        channel = frames[:, self.audio_config.channel_index]
                    except ValueError:
                        logger.warning("Reshape failed, falling back to mono interpretation.")
                        channel = samples  # assume mono

                    self.buffer.add_chunk(channel)

            except (OSError, alsaaudio.ALSAAudioError) as e:
                logger.warning(f"Mic read failed: {e}. Will attempt to reconnect...")
                self.pcm = None
                time.sleep(5)
            except ValueError as ve:
                logger.error(f"ValueError during buffer conversion: {ve}")

        self.cleanup()

    def open_device(self):
        available_capture_devices = alsaaudio.pcms(pcmtype=alsaaudio.PCM_CAPTURE)

        logger.info(f"Available ALSA devices: {available_capture_devices}")

        # Try exact match
        if self.audio_config.device_name in available_capture_devices:
            try:
                pcm = alsaaudio.PCM(
                    type=alsaaudio.PCM_CAPTURE,
                    mode=alsaaudio.PCM_NORMAL,
                    device=self.audio_config.device_name,
                    channels=self.audio_config.channels,
                    rate=self.audio_config.sample_rate,
                    format=alsaaudio.PCM_FORMAT_S16_LE,
                    periodsize=self.audio_config.period_size
                )
                logger.info(f"Connected to '{self.audio_config.device_name}'")
                return pcm
            except alsaaudio.ALSAAudioError as e:
                logger.warning(f"Failed to reopen device '{self.audio_config.device_name}': {e}")
                return None

        logger.error(f"Device '{self.audio_config.device_name}' not found and no fallback match available.")
        return None

    def pause_audio_stream(self):
        self.should_listen = False
        logger.info("Audio stream paused (not listening)")

    def resume_audio_stream(self):
        self.should_listen = True
        logger.info("Audio stream resumed (listening)")

    def stop(self):
        self.running = False

    def cleanup(self):
        if self.pcm:
            try:
                self.pcm.close()
            except Exception:
                pass
        logger.info("Mic watcher stopped cleanly.")
    
    def exit(self):
        self.cleanup()
        self.stop()
        self.join()


# === SAVE AUDIO ===
def save_audio_data(filename, audio_data, sample_rate):
    audio_data_int16 = audio_data.astype(np.int16)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data_int16.tobytes())
    logger.info(f"Audio data saved to {filename}")
    return audio_data_int16


# === ENTRY POINT ===
if __name__ == "__main__":

    audio_config = AudioConfig(device_name='hw:CARD=ArrayUAC10,DEV=0', channels=6, channel_index=0)
    buffer = AudioBuffer()
    mic_watcher = AudioManager(audio_config=audio_config, audio_buffer=buffer)
    pause_audio_stream = mic_watcher.pause_audio_stream
    resume_audio_stream = mic_watcher.resume_audio_stream
    stop_recording = mic_watcher.stop

    print(f"ðŸ“¢ Using device: {audio_config.device_name}")
    logger.info("Starting mic watcher thread...")
    mic_watcher.start()

    try:
        logger.info("â³ Recording for 5 seconds...")
        resume_audio_stream()
        time.sleep(15)
        pause_audio_stream()

        audio_data = buffer.get_and_clear(min_chunk=1024)
        save_audio_data("test_output.wav", audio_data, sample_rate=audio_config.sample_rate)

    except KeyboardInterrupt:
        pass
    finally:
        stop_recording()
        mic_watcher.join()