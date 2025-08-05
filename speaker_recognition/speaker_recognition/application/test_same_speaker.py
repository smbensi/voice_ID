from speaker_recognition.audio.audio_from_file import get_audio_chunks
from speaker_recognition.pipeline.speaker_verification import SpeakerVerificationPipeline
from speaker_recognition.utils import LOGGER

audio_file = "/home/mat/Documents/voice_ID/data/trump1_mono.mp3"
reference_audio = "/home/mat/Documents/voice_ID/data/trump2_mono.mp3"
# reference_audio = "/home/mat/Documents/voice_ID/data/huberman_3_mono.mp3"
pipe = SpeakerVerificationPipeline(model_name="speakerverification_speakernet")

chunks = get_audio_chunks(source="file", file_path=audio_file, chunk_duration=10)
print(f"{type(chunks)=}, {type(chunks[0])=}, {chunks[0].shape=}")

mean_similarity = 0

for i, chunk in enumerate(chunks):
    similarity_score = pipe.similar_speakers(chunk, reference_audio)
    LOGGER.info(f"Chunk {i} verification result: {similarity_score}")
    mean_similarity += similarity_score

mean_similarity /= len(chunks)
LOGGER.info(f"Mean similarity score: {mean_similarity}")