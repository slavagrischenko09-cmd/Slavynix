import threading
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from audio_processing import AudioSegment, assemble_audio, read_audio_segment, write_wav
from diarization import (
    DiarizationConfig,
    filter_segments_by_speaker,
    list_speakers,
    match_speaker_by_reference,
    run_diarization,
    compute_speaker_embeddings,
)
from rvc_inference import RVCConfig, run_rvc_inference
from video_processing import extract_audio, mux_audio_to_video


@dataclass
class AppState:
    video_path: str | None = None
    model_dir: str | None = None
    reference_audio: str | None = None
    target_speaker_label: str | None = None
    character_name: str | None = None
    language: str | None = None
    rvc_script: str | None = None


class VoiceSwapGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Slavynix — RVC Character Redub")

        self.state = AppState()
        self.progress_var = tk.StringVar(value="Готово к работе")

        self._build_ui()

    def _build_ui(self) -> None:
        padding = {"padx": 8, "pady": 4}

        ttk.Button(self.root, text="Загрузить фильм", command=self.select_video).grid(
            row=0, column=0, **padding
        )
        self.video_label = ttk.Label(self.root, text="Фильм не выбран")
        self.video_label.grid(row=0, column=1, sticky="w", **padding)

        ttk.Button(self.root, text="Загрузить RVC модель", command=self.select_model).grid(
            row=1, column=0, **padding
        )
        self.model_label = ttk.Label(self.root, text="Модель не выбрана")
        self.model_label.grid(row=1, column=1, sticky="w", **padding)

        ttk.Button(self.root, text="Референс голоса (опц.)", command=self.select_reference).grid(
            row=2, column=0, **padding
        )
        self.ref_label = ttk.Label(self.root, text="Референс не выбран")
        self.ref_label.grid(row=2, column=1, sticky="w", **padding)

        ttk.Button(self.root, text="Выбрать персонажа", command=self.pick_speaker).grid(
            row=3, column=0, **padding
        )
        self.speaker_label = ttk.Label(self.root, text="Speaker label не выбран")
        self.speaker_label.grid(row=3, column=1, sticky="w", **padding)

        ttk.Label(self.root, text="Имя персонажа:").grid(row=4, column=0, **padding)
        self.character_entry = ttk.Entry(self.root)
        self.character_entry.grid(row=4, column=1, sticky="ew", **padding)

        ttk.Label(self.root, text="Язык оригинала:").grid(row=5, column=0, **padding)
        self.language_entry = ttk.Entry(self.root)
        self.language_entry.grid(row=5, column=1, sticky="ew", **padding)

        ttk.Label(self.root, text="RVC script path:").grid(row=6, column=0, **padding)
        self.rvc_script_entry = ttk.Entry(self.root)
        self.rvc_script_entry.grid(row=6, column=1, sticky="ew", **padding)

        ttk.Label(self.root, text="Pitch shift:").grid(row=7, column=0, **padding)
        self.pitch_entry = ttk.Entry(self.root)
        self.pitch_entry.insert(0, "0")
        self.pitch_entry.grid(row=7, column=1, sticky="w", **padding)

        ttk.Label(self.root, text="Index rate:").grid(row=8, column=0, **padding)
        self.index_rate_entry = ttk.Entry(self.root)
        self.index_rate_entry.insert(0, "0.5")
        self.index_rate_entry.grid(row=8, column=1, sticky="w", **padding)

        ttk.Label(self.root, text="F0 method:").grid(row=9, column=0, **padding)
        self.f0_entry = ttk.Entry(self.root)
        self.f0_entry.insert(0, "rmvpe")
        self.f0_entry.grid(row=9, column=1, sticky="w", **padding)

        ttk.Button(self.root, text="Начать переозвучку", command=self.start).grid(
            row=10, column=0, **padding
        )

        ttk.Label(self.root, textvariable=self.progress_var).grid(
            row=10, column=1, sticky="w", **padding
        )

        self.root.columnconfigure(1, weight=1)

    def select_video(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mkv *.mov")])
        if path:
            self.state.video_path = path
            self.video_label.config(text=Path(path).name)

    def select_model(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.state.model_dir = path
            self.model_label.config(text=Path(path).name)

    def select_reference(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav *.mp3 *.flac")])
        if path:
            self.state.reference_audio = path
            self.ref_label.config(text=Path(path).name)

    def pick_speaker(self) -> None:
        if not self.state.video_path:
            messagebox.showerror("Ошибка", "Сначала выберите видео")
            return

        def task() -> None:
            try:
                self.progress_var.set("Диаризация...")
                workdir = self._workdir()
                audio_path = str(Path(workdir) / "audio.wav")
                extract_audio(self.state.video_path, audio_path)
                segments = run_diarization(audio_path, DiarizationConfig())
                speakers = list_speakers(segments)
                self.root.after(0, lambda: self._show_speaker_dialog(speakers))
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Ошибка", str(exc)))
            finally:
                self.progress_var.set("Готово к работе")

        threading.Thread(target=task, daemon=True).start()

    def _show_speaker_dialog(self, speakers: list[str]) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Выберите speaker label")

        listbox = tk.Listbox(dialog)
        for item in speakers:
            listbox.insert(tk.END, item)
        listbox.pack(padx=8, pady=8)

        def confirm() -> None:
            selection = listbox.curselection()
            if selection:
                label = listbox.get(selection[0])
                self.state.target_speaker_label = label
                self.speaker_label.config(text=label)
                dialog.destroy()

        ttk.Button(dialog, text="Выбрать", command=confirm).pack(pady=4)

    def start(self) -> None:
        if not self.state.video_path or not self.state.model_dir:
            messagebox.showerror("Ошибка", "Выберите видео и модель")
            return

        self.state.character_name = self.character_entry.get().strip() or "Target"
        self.state.language = self.language_entry.get().strip() or "unknown"
        self.state.rvc_script = self.rvc_script_entry.get().strip()

        if not self.state.rvc_script:
            messagebox.showerror("Ошибка", "Укажите путь к RVC-скрипту")
            return

        threading.Thread(target=self._process, daemon=True).start()

    def _process(self) -> None:
        try:
            self.progress_var.set("Извлечение аудио...")
            workdir = self._workdir()
            audio_path = str(Path(workdir) / "audio.wav")
            extract_audio(self.state.video_path, audio_path)

            self.progress_var.set("Диаризация...")
            diar_config = DiarizationConfig()
            segments = run_diarization(audio_path, diar_config)

            self.progress_var.set("Подбор персонажа...")
            speaker_label = self.state.target_speaker_label
            if self.state.reference_audio:
                embeddings = compute_speaker_embeddings(audio_path, segments, diar_config)
                speaker_label = match_speaker_by_reference(
                    self.state.reference_audio, embeddings, diar_config
                )
                self.state.target_speaker_label = speaker_label
                self.root.after(0, lambda: self.speaker_label.config(text=speaker_label))

            if not speaker_label:
                raise ValueError("Не выбран speaker label. Используйте кнопку 'Выбрать персонажа'.")

            target_segments = filter_segments_by_speaker(segments, speaker_label)

            self.progress_var.set("RVC-инференс...")
            rvc_config = self._build_rvc_config()
            processed_segments: list[AudioSegment] = []
            for idx, seg in enumerate(target_segments, start=1):
                segment_audio, sr = read_audio_segment(audio_path, seg.start, seg.end)
                input_wav = str(Path(workdir) / "segments" / f"seg_{idx:05d}_in.wav")
                output_wav = str(Path(workdir) / "segments" / f"seg_{idx:05d}_out.wav")
                write_wav(input_wav, segment_audio, sr)
                run_rvc_inference(input_wav, output_wav, rvc_config)
                processed_segments.append(
                    AudioSegment(start=seg.start, end=seg.end, speaker=seg.speaker, audio_path=output_wav)
                )
                self.progress_var.set(f"RVC-инференс: {idx}/{len(target_segments)}")

            self.progress_var.set("Сборка аудио...")
            final_audio = str(Path(workdir) / "final_audio.wav")
            assemble_audio(audio_path, processed_segments, final_audio)

            self.progress_var.set("Финальная сборка видео...")
            output_video = str(Path(workdir) / f"{Path(self.state.video_path).stem}_rvc.mp4")
            mux_audio_to_video(self.state.video_path, final_audio, output_video)

            self.progress_var.set(f"Готово: {output_video}")
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("Ошибка", str(exc)))
            self.progress_var.set("Ошибка")

    def _workdir(self) -> str:
        base = Path("outputs") / Path(self.state.video_path).stem
        base.mkdir(parents=True, exist_ok=True)
        return str(base)

    def _build_rvc_config(self) -> RVCConfig:
        model_dir = Path(self.state.model_dir)
        model_path = next(model_dir.glob("*.pth"), None)
        index_path = next(model_dir.glob("*.index"), None)
        config_path = next(model_dir.glob("*.json"), None)

        if not model_path or not index_path or not config_path:
            raise ValueError("В папке модели должны быть .pth, .index и .json файл")

        return RVCConfig(
            rvc_script_path=self.state.rvc_script,
            model_path=str(model_path),
            index_path=str(index_path),
            config_path=str(config_path),
            pitch_shift=int(self.pitch_entry.get() or 0),
            index_rate=float(self.index_rate_entry.get() or 0.5),
            f0_method=self.f0_entry.get() or "rmvpe",
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceSwapGUI(root)
    root.mainloop()
