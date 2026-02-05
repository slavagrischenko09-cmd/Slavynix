from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from pyannote.audio import Audio, Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment

from audio_processing import AudioSegment


@dataclass
class DiarizationConfig:
    hf_token: Optional[str] = None
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"


def run_diarization(audio_path: str, config: DiarizationConfig) -> List[AudioSegment]:
    pipeline = Pipeline.from_pretrained(config.diarization_model, use_auth_token=config.hf_token)
    diarization = pipeline(audio_path)

    segments: List[AudioSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(AudioSegment(start=turn.start, end=turn.end, speaker=speaker))
    return segments


def compute_speaker_embeddings(
    audio_path: str, segments: List[AudioSegment], config: DiarizationConfig
) -> Dict[str, np.ndarray]:
    embedding_model = PretrainedSpeakerEmbedding(config.embedding_model)
    audio = Audio()
    speaker_vectors: Dict[str, List[np.ndarray]] = {}

    for seg in segments:
        waveform, sample_rate = audio.crop(audio_path, Segment(seg.start, seg.end))
        vector = embedding_model(waveform[None])
        speaker_vectors.setdefault(seg.speaker, []).append(vector.squeeze(0).cpu().numpy())

    return {speaker: np.mean(vectors, axis=0) for speaker, vectors in speaker_vectors.items()}


def match_speaker_by_reference(
    reference_audio_path: str, speaker_embeddings: Dict[str, np.ndarray], config: DiarizationConfig
) -> str:
    embedding_model = PretrainedSpeakerEmbedding(config.embedding_model)
    audio = Audio()
    waveform, _ = audio.crop(reference_audio_path, Segment(0, audio.get_duration(reference_audio_path)))
    ref_vector = embedding_model(waveform[None]).squeeze(0).cpu().numpy()

    best_speaker = None
    best_score = -1.0
    for speaker, vector in speaker_embeddings.items():
        score = np.dot(ref_vector, vector) / (np.linalg.norm(ref_vector) * np.linalg.norm(vector))
        if score > best_score:
            best_score = score
            best_speaker = speaker

    if best_speaker is None:
        raise ValueError("Unable to match speaker from reference audio")
    return best_speaker


def list_speakers(segments: List[AudioSegment]) -> List[str]:
    return sorted({seg.speaker for seg in segments})


def filter_segments_by_speaker(segments: List[AudioSegment], speaker: str) -> List[AudioSegment]:
    return [seg for seg in segments if seg.speaker == speaker]
