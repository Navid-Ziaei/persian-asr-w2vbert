import io
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16000

# Manifest 'file' columns store paths relative to the processed-audio root
# (e.g. 'pool/cv/xxx.wav'). Training/eval are launched from the project root,
# but any caller in a different cwd would see LibsndfileError — resolve here.
_DEFAULT_AUDIO_ROOT = Path("data/processed")


def resolve_audio_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute() and p.exists():
        return str(p)
    if p.exists():
        return str(p)
    candidate = _DEFAULT_AUDIO_ROOT / p
    if candidate.exists():
        return str(candidate)
    # last resort: also try an absolute-ish project-root guess next to cwd parent
    alt = Path.cwd() / "data" / "processed" / p
    if alt.exists():
        return str(alt)
    return str(p)  # fall through and let soundfile produce the real error


def load_audio(path: str, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    audio, sr = sf.read(resolve_audio_path(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return audio


def load_bytes(data: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return audio