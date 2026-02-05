import subprocess
from pathlib import Path


def extract_audio(video_path: str, output_wav: str, sample_rate: int = 44100) -> None:
    """Extract audio track from video using ffmpeg."""
    output_path = Path(output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def mux_audio_to_video(video_path: str, audio_path: str, output_video: str) -> None:
    """Replace audio stream in video without re-encoding video stream."""
    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
