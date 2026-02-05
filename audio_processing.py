from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import soundfile as sf


@dataclass
class AudioSegment:
    start: float
    end: float
    speaker: str
    audio_path: str | None = None


@dataclass
class AudioMeta:
    sample_rate: int
    channels: int


def read_audio_segment(path: str, start: float, end: float) -> Tuple[np.ndarray, int]:
    """Read a segment from WAV without loading the whole file."""
    with sf.SoundFile(path) as f:
        sample_rate = f.samplerate
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        f.seek(start_frame)
        frames = end_frame - start_frame
        data = f.read(frames, dtype="float32")
    if data.ndim == 1:
        data = data[:, None]
    return data, sample_rate


def write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)


def get_audio_meta(path: str) -> AudioMeta:
    with sf.SoundFile(path) as f:
        return AudioMeta(sample_rate=f.samplerate, channels=f.channels)


def assemble_audio(
    original_audio_path: str,
    segments: List[AudioSegment],
    output_path: str,
    block_seconds: float = 10.0,
) -> None:
    """Assemble final audio while replacing only selected segments.

    Uses chunk processing to avoid loading long files into memory.
    """
    segment_cache: Dict[str, np.ndarray] = {}

    with sf.SoundFile(original_audio_path) as src:
        sample_rate = src.samplerate
        channels = src.channels
        block_frames = int(block_seconds * sample_rate)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with sf.SoundFile(output_path, mode="w", samplerate=sample_rate, channels=channels) as dst:
            total_frames = len(src)
            current_frame = 0

            sorted_segments = sorted(segments, key=lambda s: s.start)
            seg_index = 0

            while current_frame < total_frames:
                frames_to_read = min(block_frames, total_frames - current_frame)
                block = src.read(frames_to_read, dtype="float32")
                if block.ndim == 1:
                    block = block[:, None]

                block_start = current_frame / sample_rate
                block_end = (current_frame + frames_to_read) / sample_rate

                while seg_index < len(sorted_segments):
                    seg = sorted_segments[seg_index]
                    if seg.end <= block_start:
                        seg_index += 1
                        continue
                    if seg.start >= block_end:
                        break

                    if seg.audio_path is None:
                        seg_index += 1
                        continue

                    if seg.audio_path not in segment_cache:
                        seg_audio, seg_sr = read_audio_segment(seg.audio_path, 0, seg.end - seg.start)
                        if seg_sr != sample_rate:
                            raise ValueError("Sample rate mismatch in processed segment")
                        segment_cache[seg.audio_path] = seg_audio

                    seg_audio = segment_cache[seg.audio_path]
                    overlap_start = max(seg.start, block_start)
                    overlap_end = min(seg.end, block_end)
                    if overlap_end <= overlap_start:
                        seg_index += 1
                        continue

                    seg_offset = int((overlap_start - seg.start) * sample_rate)
                    block_offset = int((overlap_start - block_start) * sample_rate)
                    overlap_frames = int((overlap_end - overlap_start) * sample_rate)

                    block[block_offset : block_offset + overlap_frames] = seg_audio[
                        seg_offset : seg_offset + overlap_frames
                    ]
                    if seg.end <= block_end:
                        seg_index += 1

                dst.write(block)
                current_frame += frames_to_read
