"""Чтение и запись звука."""

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from vocoder.constants import SAMPLE_RATE


def load_audio(path: str | Path, normalize_db: float | None = None) -> np.ndarray:
    """Моно, SAMPLE_RATE, float64. При ``normalize_db`` RMS приводится к этому уровню (dBFS)."""
    x, fs = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if fs != SAMPLE_RATE:
        g = gcd(fs, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, fs // g)
    if normalize_db is not None:
        rms = np.sqrt(np.mean(x**2))
        if rms > 0:
            x = x * 10 ** (normalize_db / 20) / rms
    return x


def save_audio(path: str | Path, x: np.ndarray) -> None:
    """Сохраняет сигнал в wav, 16 бит, SAMPLE_RATE."""
    sf.write(path, x, SAMPLE_RATE, subtype="PCM_16")
