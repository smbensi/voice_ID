from speaker_recognition.pipeline.speaker_verification import SpeakerVerificationPipeline


model_name = "titanet_small"
pipe = SpeakerVerificationPipeline(model_name=model_name)

audio_files = [
    "/home/mat/Documents/voice_ID/data/jake/jake_signature.wav",
]