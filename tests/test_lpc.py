import numpy as np
from scipy.linalg import solve_toeplitz

from vocoder.analysis.lpc import autocorrelation, levinson_durbin, lpc_to_rc, prediction_gain, rc_to_lpc


def _signal(seed=0, n=400):
    """Окрашенный шум - тестовый сигнал с непустой огибающей спектра."""
    rng = np.random.default_rng(seed)
    return np.convolve(rng.standard_normal(n), [1.0, 0.8, 0.3], mode="same")


def test_levinson_matches_direct_solution():
    """Левинсон - Дарбин совпадает с прямым решением тёплицевой системы."""
    r = autocorrelation(_signal() * np.hamming(400), 10)
    a, k, err = levinson_durbin(r, 10)
    expected = solve_toeplitz(r[:10], -r[1:11])
    np.testing.assert_allclose(a[1:], expected, rtol=1e-8, atol=1e-10)
    assert np.all(np.abs(k) < 1)
    np.testing.assert_allclose(err, r[0] * prediction_gain(k))


def test_rc_lpc_roundtrip():
    """RC -> LPC -> RC возвращает исходные коэффициенты."""
    k = np.array([-0.9, 0.5, -0.3, 0.2, 0.1, -0.05, 0.3, -0.2, 0.1, 0.05])
    np.testing.assert_allclose(lpc_to_rc(rc_to_lpc(k)), k, atol=1e-12)


def test_levinson_rc_consistent_with_step_up():
    """Коэффициенты отражения из Левинсона дают тот же A(z) через step-up."""
    r = autocorrelation(_signal(1) * np.hamming(400), 10)
    a, k, _ = levinson_durbin(r, 10)
    np.testing.assert_allclose(rc_to_lpc(k), a, atol=1e-12)
