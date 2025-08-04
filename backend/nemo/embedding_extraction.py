# pre-trained model downloaded in /home/mat/.cache/torch/NeMo/NeMo_2.0.0rc0/speakerverification_speakernet/a8330fa516557b963a89ccbf0fcbe2f2/speakerverification_speakernet.nemo


from nemo.collections.asr.models import EncDecSpeakerLabelModel
from omegaconf import OmegaConf
import time

model_path = 'speakerverification_speakernet' 
model_path = 'titanet_small' # Replace with your model path
if model_path.endswith('.nemo'):
    speaker_model = EncDecSpeakerLabelModel.restore_from(model_path)
else:
    # options: 'titanet_large' (25.3M params), 
    # 'titanet_small' (6.4M params),
    # 'speakerverification_speakernet' (5M params),
    # 'ecapa_tdnn' (22.3M params)
    speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_path)

start = time.time()
embs = speaker_model.get_embedding('/home/mat/Documents/voice_ID/trump2_1_mono.mp3')
end = time.time()
print(f"Time taken to extract embedding: {end - start} seconds")
print(f"Embedding shape: {embs.shape} and type: {embs.dtype}")


audio1 = "trump1_2_mono.mp3"
audio2 = "trump2_1_mono.mp3"
# speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
decision = speaker_model.verify_speakers(audio1,audio2)
print(f"Decision:{audio1}, {audio2} {decision}")


audio1 = "huberman_3_mono.mp3"
audio2 = "trump1_2_mono.mp3"
# speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
decision = speaker_model.verify_speakers(audio1,audio2)
print(f"Decision: {audio1}, {audio2} {decision}")

'''
function verify speaker in NeMo

 @torch.no_grad()
    def verify_speakers(self, path2audio_file1, path2audio_file2, threshold=0.7):
        """
        Verify if two audio files are from the same speaker or not.

        Args:
            path2audio_file1: path to audio wav file of speaker 1
            path2audio_file2: path to audio wav file of speaker 2
            threshold: cosine similarity score used as a threshold to distinguish two embeddings (default = 0.7)

        Returns:
            True if both audio files are from same speaker, False otherwise
        """
        embs1 = self.get_embedding(path2audio_file1).squeeze()
        embs2 = self.get_embedding(path2audio_file2).squeeze()
        # Length Normalize
        X = embs1 / torch.linalg.norm(embs1)
        Y = embs2 / torch.linalg.norm(embs2)
        # Score
        similarity_score = torch.dot(X, Y) / ((torch.dot(X, X) * torch.dot(Y, Y)) ** 0.5)
        similarity_score = (similarity_score + 1) / 2

        # Decision
        if similarity_score >= threshold:
            logging.info(" two audio files are from same speaker")
            return True
        else:
            logging.info(" two audio files are from different speakers")
            return False

'''