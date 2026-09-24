"""Определение основного тона.

Все детекторы реализуют протокол :class:`PitchDetector`, чтобы их можно было заменять.
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from vocoder.constants import MAX_PERIOD, MIN_PERIOD


@dataclass
class PitchEstimate:
    period: float  # в отсчётах, с дробной частью
    aperiodicity: float  # 0 - строго периодичный сигнал, ~1 - шум
    # другие заметные провалы d'(tau): (период, значение d'), по возрастанию d'
    candidates: list[tuple[float, float]] = field(default_factory=list)


class PitchDetector(Protocol):
    window_len: int  # сколько отсчётов нужно на вход (окно центрируется на кадре)

    def estimate(self, x: np.ndarray) -> PitchEstimate:
        """Оценивает период основного тона по окну сигнала длиной window_len."""
        ...


class YinPitchDetector:
    """YIN (de Cheveigne, Kawahara, 2002) с симметричной разностной функцией.

    В исходном YIN сравнивается начало окна с его сдвинутой копией, и оценка фактически
    относится к моменту раньше центра окна. Здесь для каждого сдвига tau сравниваются
    отрезки, расположенные симметрично относительно центра окна, поэтому оценка
    привязана именно к центру.
    """

    def __init__(
        self,
        min_period: int = MIN_PERIOD,
        max_period: int = MAX_PERIOD,
        integration_len: int = 180,
        threshold: float = 0.15,
        centered: bool = True,
    ):
        """Задаёт диапазон периодов, длину окна суммирования и порог выбора провала.

        centered=False - исходный (несимметричный) вариант YIN, оставлен для сравнения.
        """
        self.min_period = min_period
        self.max_period = max_period
        self.integration_len = integration_len
        self.threshold = threshold
        self.window_len = integration_len + max_period + 1

        # Индексы пар отсчётов для каждого сдвига tau = 0..max_period+1, центрированных в окне.
        taus = np.arange(max_period + 2)
        starts = (self.window_len - integration_len - taus) // 2 if centered else np.zeros_like(taus)
        self._ia = starts[:, None] + np.arange(integration_len)
        self._ib = self._ia + taus[:, None]

    def difference(self, x: np.ndarray) -> np.ndarray:
        """d(tau) = sum (x[c+j-tau/2] - x[c+j+tau/2])^2 для tau = 0..max_period+1 (c - центр окна)."""
        return np.sum((x[self._ia] - x[self._ib]) ** 2, axis=1)

    @staticmethod
    def cmnd(d: np.ndarray) -> np.ndarray:
        """Кумулятивная нормированная разностная функция d'(tau)."""
        out = np.ones_like(d)
        csum = np.cumsum(d[1:])
        tau = np.arange(1, len(d))
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = np.where(csum > 0, d[1:] * tau / csum, 1.0)
        return out

    def estimate(self, x: np.ndarray) -> PitchEstimate:
        """Находит период: первый провал d'(tau) ниже порога, иначе глобальный минимум."""
        dn = self.cmnd(self.difference(x))
        lo, hi = self.min_period, self.max_period

        below = np.nonzero(dn[lo : hi + 1] < self.threshold)[0]
        if below.size:
            # первый провал ниже порога, затем спуск до локального минимума
            tau = lo + below[0]
            while tau < hi and dn[tau + 1] < dn[tau]:
                tau += 1
        else:
            tau = lo + int(np.argmin(dn[lo : hi + 1]))

        return PitchEstimate(self._refine(dn, tau), float(dn[tau]), self._candidates(dn))

    def _candidates(self, dn: np.ndarray, max_value: float = 0.6, count: int = 5) -> list[tuple[float, float]]:
        """Локальные минимумы d'(tau) в допустимом диапазоне ниже max_value - возможные периоды."""
        lo, hi = self.min_period, self.max_period
        seg = dn[lo - 1 : hi + 2]
        idx = np.nonzero((seg[1:-1] < seg[:-2]) & (seg[1:-1] <= seg[2:]) & (seg[1:-1] < max_value))[0] + lo
        best = sorted(idx, key=lambda t: dn[t])[:count]
        return [(self._refine(dn, int(t)), float(dn[t])) for t in best]

    @staticmethod
    def _refine(dn: np.ndarray, tau: int) -> float:
        """Параболическая интерполяция минимума."""
        if tau <= 0 or tau >= len(dn) - 1:
            return float(tau)
        a, b, c = dn[tau - 1], dn[tau], dn[tau + 1]
        denom = a - 2 * b + c
        if denom <= 0:
            return float(tau)
        return tau + 0.5 * (a - c) / denom
