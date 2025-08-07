import torch
import numpy as np
import pandas as pd

from nemo.collections.asr.models import EncDecSpeakerLabelModel

from speaker_verification.settings import recognition_params

class VoiceRecognition:
    
    def __init__(self, model_name: str = "titanet_small"):
        self.speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_name=model_name)
        self.embeddings = pd.json_normalize(recognition_params.EMBEDDING_FILE)

    
    def similar_speakers(self, audio1: str, audio2: str) -> bool:
        similarity_score = self.similarity(audio1, audio2)
        return similarity_score > 0.7

    def verify_speaker(self, audio_path: str, reference_audio: str) -> float:
        pass
    
    def get_embedding_from_audio_file(self, audio_path: str):
        # mono_waveform = self.check_audio_is_mono(audio_path)
        # embs = self.infer_segment(mono_waveform)[0].squeeze()
        embs = self.speaker_model.get_embedding(audio_path).squeeze()
        return embs    
    
    def get_embedding_segment(self, audio_segment: np.ndarray):
        embs = self.infer_segment(audio_segment)[0].squeeze()
        return embs
    
    
    def is_recognized(self, audio: str | np.ndarray, reference_audio: str | list[np.ndarray] = None) -> bool:
        """
        Verify if the audio is recognized as the same speaker as the reference audios.
        
        Args:
            audio (str | np.ndarray): Path to the audio file or a numpy array of audio data.
            reference_audio (str | list[np.ndarray]): Path to the reference audio file or a list of numpy arrays representing embeddings 
        
        Returns:
            bool: True if recognized, False otherwise.
        """
        if isinstance(audio, str):
            audio_embedding = self.get_embedding_from_audio_file(audio)
        elif isinstance(audio, np.ndarray):
            audio_embedding = self.get_embedding_segment(audio)

        if reference_audio is None:
            # compare to embeddings in database
            for _, row in self.embeddings.iterrows():
                reference_embedding = torch.tensor(row['embedding'])
                similarity_score = self.compare_embeddings(audio_embedding, reference_embedding)
                if similarity_score > recognition_params.THRESHOLD:
                    return row['speaker_id'] 
        
        elif isinstance(reference_audio, str):
            reference_embedding = self.get_embedding_from_audio_file(reference_audio)
        elif isinstance(reference_audio, np.ndarray):
            reference_embedding = self.get_embedding_segment(reference_audio)

        similarity_score = self.compare_embeddings(audio_embedding, reference_embedding)
        
        return similarity_score 
    
    
    def compare_embeddings(self, embs1, embs2):
        # Length Normalize
        if embs1.ndim != 1 or embs2.ndim != 1:
            raise ValueError(f"Embeddings must be 1-dimensional arrays. {embs1.ndim=}, {embs2.ndim=}")
        X = embs1 / torch.linalg.norm(embs1)
        Y = embs2 / torch.linalg.norm(embs2)
        # Score
        similarity_score = torch.dot(X, Y) / ((torch.dot(X, X) * torch.dot(Y, Y)) ** 0.5)
        similarity_score = (similarity_score + 1) / 2
        return similarity_score
        
    def infer_segment(self, segment):
        """
        Args:
            segment: segment of audio file

        Returns:
            emb: speaker embeddings (Audio representations)
            logits: logits corresponding of final layer
        """
        segment_length = segment.shape[0]

        device = self.speaker_model.device
        # audio = np.array([segment])
        audio = segment 
        audio_signal, audio_signal_len = (
            torch.tensor(audio, device=device, dtype=torch.float32),
            torch.tensor([segment_length], device=device),
        )
        mode = self.speaker_model.training
        self.speaker_model.freeze()
        
        logits, emb = self.speaker_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)

        self.speaker_model.train(mode=mode)
        if mode is True:
            self.speaker_model.unfreeze()
        del audio_signal, audio_signal_len
        return emb, logits
