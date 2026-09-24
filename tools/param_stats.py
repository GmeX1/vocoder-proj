"""Статистика параметров кадров на обучающих записях - для подбора диапазонов квантователей.

    uv run python tools/param_stats.py data/fleurs_ru/test --exclude data/testset/manifest.tsv
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from vocoder.analysis.analyzer import Analyzer
from vocoder.audio import load_audio


def main():
    """Анализирует записи и печатает процентили параметров кадров."""
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir", type=Path)
    ap.add_argument("--exclude", type=Path, help="manifest.tsv тестового набора (столбец source)")
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()

    skip = set()
    if args.exclude:
        with open(args.exclude, encoding="utf-8") as f:
            skip = {row["source"] for row in csv.DictReader(f, delimiter="\t")}
    files = sorted(p for p in args.audio_dir.glob("*.wav") if p.name not in skip)[: args.limit]

    rc, rms, voiced = [], [], []
    for path in files:
        x = load_audio(path, normalize_db=-26.0)
        an = Analyzer()
        for fr in an.push(x) + an.flush():
            rc.append(fr.rc)
            rms.append(fr.rms)
            voiced.append(fr.voiced)
    rc, rms, voiced = np.array(rc), np.array(rms), np.array(voiced)
    speech = 20 * np.log10(rms + 1e-12) > -70

    print(f"файлов: {len(files)}, кадров: {len(rms)}, звонких: {voiced.mean():.2f}")
    db = 20 * np.log10(rms[speech])
    print(f"RMS, dB: p0.5={np.percentile(db, 0.5):.1f} p99.9={np.percentile(db, 99.9):.1f}")
    lar = 2 * np.arctanh(np.clip(rc[speech, :2], -0.9999, 0.9999))
    for i in range(2):
        print(f"LAR{i + 1}: p0.5={np.percentile(lar[:, i], 0.5):+.2f} p99.5={np.percentile(lar[:, i], 99.5):+.2f}")
    for i in range(2, rc.shape[1]):
        col = rc[speech & voiced, i]
        print(f"k{i + 1}: p0.5={np.percentile(col, 0.5):+.3f} p99.5={np.percentile(col, 99.5):+.3f}")


if __name__ == "__main__":
    main()
