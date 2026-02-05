from dataclasses import dataclass
from pathlib import Path
import subprocess


@dataclass
class RVCConfig:
    rvc_script_path: str
    model_path: str
    index_path: str
    config_path: str
    pitch_shift: int = 0
    index_rate: float = 0.5
    f0_method: str = "rmvpe"
    filter_radius: int = 3
    protect: float = 0.5
    resample_sr: int = 0


def run_rvc_inference(input_wav: str, output_wav: str, config: RVCConfig) -> None:
    """Run RVC inference via external CLI wrapper."""
    output_path = Path(output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        config.rvc_script_path,
        "--input",
        input_wav,
        "--output",
        str(output_path),
        "--model",
        config.model_path,
        "--index",
        config.index_path,
        "--config",
        config.config_path,
        "--pitch",
        str(config.pitch_shift),
        "--index-rate",
        str(config.index_rate),
        "--f0-method",
        config.f0_method,
        "--filter-radius",
        str(config.filter_radius),
        "--protect",
        str(config.protect),
        "--resample-sr",
        str(config.resample_sr),
    ]
    subprocess.run(cmd, check=True)
