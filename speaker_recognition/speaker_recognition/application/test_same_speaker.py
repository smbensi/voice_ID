import os

import torchaudio

from speaker_recognition.audio.audio_from_file import get_audio_chunks
from speaker_recognition.pipeline.speaker_verification import SpeakerVerificationPipeline
from speaker_recognition.utils import LOGGER

audio_file = "/home/mat/Documents/voice_ID/mathias_respeaker_sample.wav"
# reference_audio = "/home/mat/Documents/voice_ID/data/trump2_mono.mp3"
# reference_audio = "/home/mat/Documents/voice_ID/data/mathias_phone.mp3"
# reference_audio = "/home/mat/Documents/voice_ID/mathias_respeaker_sample.wav"
# reference_audio = "/home/mat/Documents/voice_ID/huberman/2/s2_0017_mono.wav"
# reference_audio = "/home/mat/Documents/voice_ID/data/huberman/huberman_3_mono.mp3"


reference_audio = "/home/mat/Documents/voice_ID/data/jake/jake_signature.wav"
reference_audio = "/home/mat/Documents/voice_ID/data/adina/adina.wav"


pipe = SpeakerVerificationPipeline(model_name="titanet_small")
need_chunks = False  # Set to False if you want to verify a single audio file


if need_chunks:
    chunks = get_audio_chunks(source="file", file_path=audio_file, chunk_duration=10)
    print(f"{type(chunks)=}, {type(chunks[0])=}, {chunks[0].shape=}")

def get_audio_duration(file_path):
    info = torchaudio.info(file_path)
    duration = info.num_frames / info.sample_rate
    return duration

folder_path = "//home/mat/Documents/voice_ID/data/long_audio/adina_sagi/3"
chunks = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
          if os.path.isfile(os.path.join(folder_path, f)) and 
          get_audio_duration(os.path.join(folder_path, f)) > 1]
durations = [get_audio_duration(chunk) for chunk in chunks]


mean_similarity = 0

for i, chunk in enumerate(chunks):
    similarity_score = pipe.similar_speakers(chunk, reference_audio)
    LOGGER.info(f"Chunk {chunk} with duration {durations[i]} verification result: {similarity_score:.2f}")
    mean_similarity += similarity_score

mean_similarity /= len(chunks)
LOGGER.info(f"Mean similarity score: {mean_similarity}")