"""Демонстрационное видео для README: оригинал, эталон LPC-10e и наш вокодер подряд.

    uv run python tools/make_demo_video.py m03 --title "Мужской голос" --out data/video/m03.mp4

На экране три спектрограммы друг под другом; звучащая дорожка подсвечена, по ней бежит
отметка текущего момента. Все варианты приведены к громкости оригинала. Нужен ffmpeg.

GitHub не показывает плеер для аудио в README, но показывает его для mp4, загруженного
через веб-редактор (перетаскиванием файла).
"""

import argparse
import csv
import subprocess
import tempfile
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Fira Sans", "DejaVu Sans"]  # DejaVu - запасной, если Fira Sans нет
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.animation import FFMpegWriter
from scipy.signal import spectrogram

from vocoder.constants import SAMPLE_RATE

TRACKS = [
    ("Оригинал", "data/testset"),
    ("Эталон LPC-10e, 2400 бит/с", "data/out_ref_lpc10e"),
    ("Наш вокодер, 2400 бит/с", "data/out/lpc10"),
]
LEAD, GAP, TAIL = 0.5, 0.8, 0.7  # паузы в секундах: в начале, между вариантами, в конце
FPS = 25


def load_tracks(name: str) -> list[np.ndarray]:
    """Загружает варианты записи и приводит их к громкости оригинала."""
    signals = [sf.read(Path(d) / f"{name}.wav")[0] for _, d in TRACKS]
    n = len(signals[0])
    ref_rms = np.std(signals[0])
    out = []
    for y in signals:
        y = np.pad(y[:n], (0, max(0, n - len(y))))
        out.append(y * ref_rms / (np.std(y) + 1e-12))
    return out


def manifest_row(name: str, manifest: Path = Path("data/testset/manifest.tsv")) -> dict:
    """Строка манифеста тестового набора (текст фразы и др.)."""
    with open(manifest, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["file"] == f"{name}.wav":
                return row
    return {"text": "", "source": "?"}


def build_audio(signals: list[np.ndarray]) -> tuple[np.ndarray, list[float]]:
    """Склеивает варианты с паузами; возвращает звук и моменты начала каждого варианта."""
    parts, starts, t = [np.zeros(int(LEAD * SAMPLE_RATE))], [], LEAD
    for i, y in enumerate(signals):
        starts.append(t)
        parts.append(y)
        t += len(y) / SAMPLE_RATE
        pause = GAP if i < len(signals) - 1 else TAIL
        parts.append(np.zeros(int(pause * SAMPLE_RATE)))
        t += pause
    audio = np.concatenate(parts)
    return audio / max(np.abs(audio).max() / 0.95, 1.0), starts


def render(name: str, title: str, out: Path) -> None:
    """Рисует кадры видео, кодирует их и добавляет звук."""
    signals = load_tracks(name)
    audio, starts = build_audio(signals)
    duration = len(signals[0]) / SAMPLE_RATE
    total = len(audio) / SAMPLE_RATE

    specs = []
    for y in signals:
        f, t, s = spectrogram(y, fs=SAMPLE_RATE, nperseg=256, noverlap=224)
        specs.append((f, t, 10 * np.log10(s + 1e-12)))
    vmax = specs[0][2].max()

    fig, axes = plt.subplots(len(TRACKS), 1, figsize=(12.8, 7.2), dpi=100, sharex=True)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.8, bottom=0.08, hspace=0.35)
    row = manifest_row(name)
    header = f"{title} ({name}.wav)"
    fig.suptitle(header, fontsize=18, x=0.02, ha="left", y=0.97)
    phrase = textwrap.fill(f'"{row["text"]}"', width=130)
    fig.text(0.02, 0.9, phrase, fontsize=12, color="0.25", va="top")
    images, labels, cursors = [], [], []
    for ax, (label, _), (f, t, s) in zip(axes, TRACKS, specs):
        extent = (t[0], t[-1], f[0] / 1000, f[-1] / 1000)
        images.append(ax.imshow(s, origin="lower", aspect="auto", extent=extent, cmap="magma",
                                vmin=vmax - 70, vmax=vmax, interpolation="nearest"))
        ax.set_ylabel("кГц")
        ax.set_ylim(0, 4)
        labels.append(ax.set_title(label, loc="left", fontsize=13))
        cursors.append(ax.axvline(0, color="#00e5ff", lw=2.5, visible=False))
    axes[-1].set_xlabel("время, с")
    axes[-1].set_xlim(0, duration)

    def update(time: float) -> None:
        """Подсвечивает звучащий вариант и двигает отметку времени."""
        active = max((i for i, s in enumerate(starts) if time >= s), default=-1)
        for i, (img, lab, cur) in enumerate(zip(images, labels, cursors)):
            playing = i == active and time - starts[i] <= duration
            img.set_alpha(1.0 if playing else 0.3)
            lab.set_color("black" if playing else "0.6")
            lab.set_fontweight("bold" if playing else "normal")
            cur.set_visible(playing)
            if playing:
                cur.set_xdata([time - starts[i]] * 2)

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        silent, wav = Path(tmp) / "video.mp4", Path(tmp) / "audio.wav"
        writer = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "26"])
        with writer.saving(fig, str(silent), dpi=100):
            for k in range(int(total * FPS)):
                update(k / FPS)
                writer.grab_frame()
        plt.close(fig)
        sf.write(wav, audio, SAMPLE_RATE, subtype="PCM_16")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", str(wav), "-c:v", "copy",
             "-c:a", "aac", "-ar", "44100", "-b:a", "96k", "-shortest", str(out)],
            check=True,
        )


def main():
    """Разбирает аргументы и собирает видео."""
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="имя записи тестового набора без .wav, например m03")
    ap.add_argument("--title", default="", help="подпись, например 'Мужской голос'")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    render(args.name, args.title or args.name, args.out)


if __name__ == "__main__":
    main()
