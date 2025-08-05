import torch
import torchaudio
import numpy as np


from nemo.collections.asr.models import EncDecSpeakerLabelModel

import os
import logging

# Set NeMo log level before importing NeMo
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

from nemo.collections.asr.models import EncDecSpeakerLabelModel

from speaker_recognition.utils import LOGGER

class SpeakerVerificationPipeline:
    '''
        # options: 'titanet_large' (25.3M params) -> embedding 192 dim, 
    # 'titanet_small' (6.4M params) -> embedding 192 dim,
    # 'speakerverification_speakernet' (5M params),
    # 'ecapa_tdnn' (22.3M params) -> embedding 192 dim
    '''
    def __init__(self, model_name: str = "titanet_large"):
        self.speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_name=model_name)

    def similar_speakers(self, audio1: str, audio2: str) -> bool:
        similarity_score = self.similarity(audio1, audio2)
        return similarity_score

    def check_audio_is_mono(self, audio_path: str) -> bool:

        # Load audio (returns waveform and sample rate)
        waveform, sample_rate = torchaudio.load("input.wav")  # or .mp3, .aac, etc.
        
        # Check sample rate
        if sample_rate != 16000:
            LOGGER.warning(f"Sample rate for {audio_path} is {sample_rate} Hz, expected 16000 Hz.")
            LOGGER.info(f"Resampling {audio_path} from {sample_rate} Hz to 16000 Hz.")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000


        # If not mono, convert to mono by averaging channels
        if waveform.shape[0] > 1:
            LOGGER.info(f"Converting {audio_path} to mono.")
            mono_waveform = waveform.mean(dim=0, keepdim=True)
        else:
            mono_waveform = waveform

        # Save as mono WAV
        # torchaudio.save("output_mono.wav", mono_waveform, sample_rate)
        return mono_waveform
    
    def similarity(self, path2audio_file1, path2audio_file2, threshold=0.7):
        """
        Verify if two audio files are from the same speaker or not.

        Args:
            path2audio_file1: path to audio wav file of speaker 1
            path2audio_file2: path to audio wav file of speaker 2
            threshold: cosine similarity score used as a threshold to distinguish two embeddings (default = 0.7)

        Returns:
            True if both audio files are from same speaker, False otherwise
        """
        if isinstance(path2audio_file1, str):
            embs1 = self.speaker_model.get_embedding(path2audio_file1).squeeze()
        else:
            # LOGGER.debug(f"Inferring segment for {path2audio_file1} shape {path2audio_file1.shape}")
            embs1 = self.speaker_model.infer_segment(path2audio_file1)[0].squeeze()
        LOGGER.debug(f"Embeddings 1 shape: {embs1.shape}")
        if isinstance(path2audio_file2, str):
            embs2 = self.speaker_model.get_embedding(path2audio_file2).squeeze()
        else:
            embs2 = self.speaker_model.infer_segment(path2audio_file2)[0].squeeze()
        LOGGER.debug(f"Embeddings 2 shape: {embs2.shape}")
        # Length Normalize
        X = embs1 / torch.linalg.norm(embs1)
        Y = embs2 / torch.linalg.norm(embs2)
        # Score
        similarity_score = torch.dot(X, Y) / ((torch.dot(X, X) * torch.dot(Y, Y)) ** 0.5)
        similarity_score = (similarity_score + 1) / 2
        # LOGGER.info(f"Similarity score between {path2audio_file1} and {path2audio_file2}: {similarity_score.item()}")
        return similarity_score
        
        # Decision
        # if similarity_score >= threshold:
        #     logging.info(" two audio files are from same speaker")
        #     return True
        # else:
        #     logging.info(" two audio files are from different speakers")
        #     return False
        
        
    def infer_segment(self, segment):
        """
        Args:
            segment: segment of audio file

        Returns:
            emb: speaker embeddings (Audio representations)
            logits: logits corresponding of final layer
        """
        segment_length = segment.shape[0]

        device = self.device
        # audio = np.array([segment])
        audio = segment 
        audio_signal, audio_signal_len = (
            torch.tensor(audio, device=device, dtype=torch.float32),
            torch.tensor([segment_length], device=device),
        )
        mode = self.training
        self.freeze()

        logits, emb = self.speaker_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)

        self.train(mode=mode)
        if mode is True:
            self.unfreeze()
        del audio_signal, audio_signal_len
        return emb, logits