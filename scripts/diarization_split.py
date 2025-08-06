import os
import json
from pathlib import Path
from typing import Dict, List
from pydub import AudioSegment

def split_phrases(
    audio_path: str,
    json_path: str,
    out_dir: str = "huberman",
    fmt: str = "wav",
    normalize_mono_16k: bool = False,  # set True if you want ASR-friendly mono/16 kHz
):
    audio_path = Path(audio_path)
    json_path = Path(json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load JSON
    meta: Dict = json.loads(json_path.read_text(encoding="utf-8"))
    phrases: List[Dict] = meta.get("phrases", [])
    if not phrases:
        raise ValueError("No 'phrases' array found in JSON.")

    # Load audio via pydub (milliseconds indexing)
    audio = AudioSegment.from_file(audio_path)

    # Optional: standardize to mono/16k for ASR pipelines
    if normalize_mono_16k:
        audio = audio.set_channels(1).set_frame_rate(16000)

    # Maintain per-speaker index
    counters: Dict[int, int] = {}

    for p in phrases:
        speaker = int(p.get("speaker", 0))
        out_file = out_dir / f"{speaker}"
        if not os.path.exists(out_file):
            out_file.mkdir(parents=True, exist_ok=True)
        start_ms = int(p.get("offsetMilliseconds", 0))
        dur_ms = int(p.get("durationMilliseconds", 0))
        end_ms = start_ms + dur_ms

        # Update per-speaker counter
        counters.setdefault(speaker, 0)
        counters[speaker] += 1
        idx = counters[speaker]

        # Slice and export
        seg = audio[start_ms:end_ms]
        out_name = f"s{speaker}_{idx:04d}.{fmt}"
        out_file = out_dir / f"{speaker}" /  out_name

        # Export as WAV (or chosen fmt)
        seg.export(out_file, format=fmt)
        print(f"Wrote: {out_file}  ({start_ms}–{end_ms} ms, speaker {speaker}, idx {idx})")

if __name__ == "__main__":
    # Example usage:
    # python split_phrases.py
    # Edit the paths below or wire them to argparse if you prefer CLI args.
    split_phrases(
        audio_path="/home/mat/Documents/voice_ID/data/Huberman.mp3",     # <-- your original audio file
        json_path="/home/mat/Documents/voice_ID/transcription_huberman_no_words.json",      # <-- your JSON file
        out_dir="huberman",
        fmt="wav",
        normalize_mono_16k=False       # set True if you want mono/16k output
    )
