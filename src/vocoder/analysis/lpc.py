"""Линейное предсказание.

Соглашение о знаках: полином фильтра анализа A(z) = 1 + a1*z^-1 + ... + ap*z^-p,
то есть массив ``a`` имеет вид [1, a1, ..., ap] и подходит для ``scipy.signal.lfilter``.
"""

import numpy as np


def autocorrelation(x: np.ndarray, order: int) -> np.ndarray:
    """Автокорреляция R[0..order] (без нормировки)."""
    n = len(x)
    return np.array([np.dot(x[: n - i], x[i:]) for i in range(order + 1)])


def levinson_durbin(r: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Решает нормальные уравнения рекурсией Левинсона - Дарбина.

    Возвращает (a, k, err): коэффициенты полинома A(z), коэффициенты отражения
    k1..kp и энергию ошибки предсказания.
    """
    a = np.zeros(order + 1)
    a[0] = 1.0
    k = np.zeros(order)
    err = r[0]
    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i - 1 : 0 : -1])
        ki = -acc / err
        k[i - 1] = ki
        a[1:i] = a[1:i] + ki * a[i - 1 : 0 : -1]
        a[i] = ki
        err *= 1.0 - ki * ki
    return a, k, err


def rc_to_lpc(k: np.ndarray) -> np.ndarray:
    """Коэффициенты отражения -> полином A(z) (step-up рекурсия)."""
    a = np.array([1.0])
    for ki in k:
        a = np.concatenate([a, [0.0]])
        a = a + ki * a[::-1]
    return a


def lpc_to_rc(a: np.ndarray) -> np.ndarray:
    """Полином A(z) -> коэффициенты отражения (step-down рекурсия)."""
    a = np.asarray(a, dtype=float).copy()
    p = len(a) - 1
    k = np.zeros(p)
    for i in range(p, 0, -1):
        ki = a[i]
        k[i - 1] = ki
        if i > 1:
            a = (a[:i] - ki * a[i:0:-1]) / (1.0 - ki * ki)
    return k


def prediction_gain(k: np.ndarray) -> float:
    """prod(1 - ki^2): доля энергии сигнала, остающаяся в ошибке предсказания."""
    return float(np.prod(1.0 - np.asarray(k) ** 2))
