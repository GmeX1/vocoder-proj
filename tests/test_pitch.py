import numpy as np
import pytest
from scipy.signal import lfilter

from vocoder.analysis.pitch import YinPitchDetector


def _voiced(f0, n, fs=8000):
    """Синтетический звонкий сигнал: импульсы с частотой f0 через резонансный фильтр."""
    exc = np.zeros(n)
    exc[:: int(round(fs / f0))] = 1.0
    return lfilter([1.0], [1.0, -1.3, 0.8], exc)  # простой "формантный" фильтр


@pytest.mark.parametrize("f0", [80, 120, 200, 320])
def test_yin_finds_period(f0):
    """YIN находит период с точностью 2% и признаёт сигнал периодичным."""
    det = YinPitchDetector()
    est = det.estimate(_voiced(f0, det.window_len))
    assert est.period == pytest.approx(8000 / f0, rel=0.02)
    assert est.aperiodicity < 0.15


def test_yin_noise_is_aperiodic():
    """Для белого шума апериодичность высокая."""
    det = YinPitchDetector()
    est = det.estimate(np.random.default_rng(0).standard_normal(det.window_len))
    assert est.aperiodicity > 0.5
