import numpy as np

from vocoder.analysis.voicing import LevelTracker, VoicingDetector, VoicingFeatures


def _feats(n, seed=0):
    """Случайная последовательность признаков полукадров."""
    rng = np.random.default_rng(seed)
    return [
        VoicingFeatures(
            level_db=float(rng.uniform(-70, -20)),
            aperiodicity=float(rng.uniform(0, 1)),
            zcr=float(rng.uniform(0, 0.6)),
            low_ratio=float(rng.uniform(0, 1)),
            k1=float(rng.uniform(-1, 0.5)),
        )
        for _ in range(n)
    ]


def test_one_decision_per_half_frame():
    """Каждый полукадр получает ровно одно решение, включая последние после flush."""
    for n in (1, 2, 3, 10):
        assert len(VoicingDetector().decide_all(_feats(n))) == n


def test_decisions_delayed_by_two_half_frames():
    """До flush решения отстают от входа ровно на DELAY полукадров."""
    det = VoicingDetector()
    out = []
    for i, f in enumerate(_feats(12)):
        out += det.push(f)
        assert len(out) == max(0, i + 1 - VoicingDetector.DELAY)


def test_silence_is_unvoiced():
    """Полукадры тише порога тишины всегда глухие, даже если сигнал периодичен."""
    quiet = [VoicingFeatures(level_db=-80, aperiodicity=0.0, zcr=0.05, low_ratio=0.9, k1=-0.9) for _ in range(5)]
    assert not any(VoicingDetector().decide_all(quiet))


def test_level_tracker_relative_to_peak():
    """Уровень считается от пика; пик медленно спадает."""
    t = LevelTracker(decay_db=1.0, floor_db=-100)
    assert t.relative(-20) == 0
    assert t.relative(-30) == -9  # пик спал на 1 дБ
    assert t.relative(-10) == 0
