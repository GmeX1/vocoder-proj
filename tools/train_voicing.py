"""Обучение классификатора "звонкий / глухой" по разметке Praat.

    uv run python tools/train_voicing.py data/fleurs_ru/test --exclude data/testset/manifest.tsv \\
        --eval data/testset

Для каждой половины кадра считаются признаки анализатора (с контекстом, как в
VoicingDetector), а метка "звонкий" берётся из Praat (есть ли у него тон в центре
половины кадра). На признаках обучается логистическая регрессия; её веса переносятся
в VoicingDetector.WEIGHTS / BIAS.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import parselmouth

from vocoder.analysis.analyzer import Analyzer
from vocoder.analysis.pitch import YinPitchDetector
from vocoder.analysis.voicing import LevelTracker, VoicingDetector, VoicingFeatures
from vocoder.audio import load_audio
from vocoder.constants import FRAME_LEN, SAMPLE_RATE


def praat_voiced(x: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Метки Praat: есть ли тон в заданные моменты времени (секунды)."""
    pitch = parselmouth.Sound(x, SAMPLE_RATE).to_pitch_ac(time_step=0.005, pitch_floor=60, pitch_ceiling=400)
    return np.array([not np.isnan(pitch.get_value_at_time(t)) for t in times])


def context_matrix(feats: list[VoicingFeatures]) -> np.ndarray:
    """Векторы признаков с контекстом - так же, как их строит VoicingDetector."""
    tracker = LevelTracker()
    ap = [f.aperiodicity for f in feats]
    rows = []
    for i, f in enumerate(feats):
        prev_ap = ap[i - 1] if i > 0 else ap[i]
        next_ap = ap[i + 1] if i + 1 < len(ap) else ap[i]
        rows.append(VoicingDetector.vector(f, tracker.relative(f.level_db), prev_ap, next_ap))
    return np.array(rows)


def collect(files: list[Path], detector: VoicingDetector | None = None, classic_yin: bool = False, normalize=None):
    """Признаки, уровни, метки Praat и решения детектора для всех половин кадров."""
    X, level, y, pred = [], [], [], []
    for path in files:
        x = load_audio(path, normalize_db=normalize)
        an = Analyzer(pitch_detector=YinPitchDetector(centered=not classic_yin), voicing=detector)
        an.feature_log = []
        frames = an.push(x) + an.flush()
        n_half = 2 * len(frames)
        times = (np.arange(n_half) * FRAME_LEN / 2 + FRAME_LEN / 4) / SAMPLE_RATE
        X.append(context_matrix(an.feature_log))
        level += [f.level_db for f in an.feature_log]
        y.append(praat_voiced(x, times))
        pred += [v for fr in frames for v in fr.voicing]
    return np.concatenate(X), np.array(level), np.concatenate(y), np.array(pred)


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 50) -> tuple[np.ndarray, float]:
    """Логистическая регрессия методом Ньютона с балансировкой классов и L2-регуляризацией."""
    A = np.hstack([X, np.ones((len(X), 1))])
    sw = np.where(y, 0.5 / y.mean(), 0.5 / (1 - y.mean()))  # веса, уравнивающие классы
    w = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ w))
        grad = A.T @ (sw * (p - y)) + l2 * w
        H = (A * (sw * p * (1 - p))[:, None]).T @ A + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, grad)
    return w[:-1], float(w[-1])


def report(name: str, y: np.ndarray, pred: np.ndarray) -> None:
    """Печатает доли ошибок в обе стороны."""
    print(
        f"{name}: точность {np.mean(y == pred):.3f}, звонкий->глухой {np.mean(~pred[y]):.3f}, "
        f"глухой->звонкий {np.mean(pred[~y]):.3f}"
    )


def main():
    """Собирает признаки, обучает классификатор и проверяет его на тестовом наборе."""
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir", type=Path)
    ap.add_argument("--exclude", type=Path, help="manifest.tsv тестового набора (столбец source)")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--eval", type=Path, help="папка тестового набора для проверки")
    ap.add_argument("--classic-yin", action="store_true", help="несимметричный YIN (для сравнения)")
    args = ap.parse_args()

    skip = set()
    if args.exclude:
        with open(args.exclude, encoding="utf-8") as f:
            skip = {row["source"] for row in csv.DictReader(f, delimiter="\t")}
    files = sorted(p for p in args.audio_dir.glob("*.wav") if p.name not in skip)[: args.limit]

    X, level, y, _ = collect(files, classic_yin=args.classic_yin, normalize=-26.0)
    speech = level >= -60.0
    weights, bias = fit_logistic(X[speech], y[speech])
    print(f"обучено на {speech.sum()} половинах кадров, доля звонких {y[speech].mean():.2f}")
    print(f"WEIGHTS = np.array([{', '.join(f'{v:.3f}' for v in weights)}])")
    print(f"BIAS = {bias:.3f}")

    if args.eval:
        for g in "mf":
            test = sorted(args.eval.glob(f"{g}*.wav"))
            for name, det in [("старый", None), ("новый", VoicingDetector(weights, bias))]:
                _, _, yt, pt = collect(test, det, classic_yin=args.classic_yin)
                report(f"{'мужчины' if g == 'm' else 'женщины'}, {name}", yt, pt)


if __name__ == "__main__":
    main()
