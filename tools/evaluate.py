"""Оценка вокодера на тестовом наборе: PESQ (узкополосный), STOI, скорость работы.

    uv run python tools/evaluate.py data/testset --out data/out

Варианты:
  unquantized - анализ -> синтез без квантования (потолок качества для модели);
  lpc10       - полный путь через битовый поток 2400 бит/с.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from pesq import pesq
from pystoi import stoi

from vocoder.analysis.analyzer import Analyzer
from vocoder.audio import load_audio, save_audio
from vocoder.codec import decode_bytes, encode_signal
from vocoder.constants import SAMPLE_RATE
from vocoder.synthesis.synthesizer import Synthesizer


def run_unquantized(x: np.ndarray) -> np.ndarray:
    """Анализ -> синтез без квантования."""
    an, syn = Analyzer(), Synthesizer()
    y = np.concatenate([syn.synthesize(f) for f in an.push(x) + an.flush()] + [syn.flush()])
    return y[syn.delay :]


def run_lpc10(x: np.ndarray) -> np.ndarray:
    """Полный путь через битовый поток 2400 бит/с."""
    return decode_bytes(encode_signal(x))


VARIANTS = {"unquantized": run_unquantized, "lpc10": run_lpc10}


def main():
    """Прогоняет варианты по тестовому набору и печатает средние метрики."""
    ap = argparse.ArgumentParser()
    ap.add_argument("testset", type=Path)
    ap.add_argument("--out", type=Path, help="куда сохранить синтезированные wav")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    args = ap.parse_args()

    files = sorted(args.testset.glob("*.wav"))
    for name in args.variants:
        scores, cpu, dur = [], 0.0, 0.0
        for path in files:
            x = load_audio(path)
            t = time.perf_counter()
            y = VARIANTS[name](x)
            cpu += time.perf_counter() - t
            dur += len(x) / SAMPLE_RATE
            y = np.pad(y[: len(x)], (0, max(0, len(x) - len(y))))
            scores.append((path.stem[0], pesq(SAMPLE_RATE, x, y, "nb"), stoi(x, y, SAMPLE_RATE)))
            if args.out:
                (args.out / name).mkdir(parents=True, exist_ok=True)
                save_audio(args.out / name / path.name, y)

        s = np.array([(p, q) for _, p, q in scores])
        by_sex = {g: s[[i for i, (sx, _, _) in enumerate(scores) if sx == g]].mean(axis=0) for g in "mf"}
        print(
            f"{name:12s} PESQ {s[:, 0].mean():.2f} (м {by_sex['m'][0]:.2f} / ж {by_sex['f'][0]:.2f})  "
            f"STOI {s[:, 1].mean():.3f} (м {by_sex['m'][1]:.3f} / ж {by_sex['f'][1]:.3f})  "
            f"время/длительность {cpu / dur:.3f}"
        )


if __name__ == "__main__":
    main()
