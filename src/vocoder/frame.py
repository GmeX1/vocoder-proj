"""Параметры одного кадра речи - то, что передаётся от кодера к декодеру."""

from dataclasses import dataclass, field

import numpy as np

from vocoder.constants import LPC_ORDER


@dataclass
class FrameParams:
    """Кадр делится на две половины: вокализованность и громкость задаются для каждой.

    Огибающая спектра (rc) и период тона - одни на весь кадр.
    """

    voicing: tuple[bool, bool]  # звонкая ли первая / вторая половина кадра
    period: float  # период основного тона в отсчётах; 0, если обе половины глухие
    gains: tuple[float, float]  # RMS первой / второй половины после предыскажения
    rc: np.ndarray = field(default_factory=lambda: np.zeros(LPC_ORDER))  # коэффициенты отражения k1..kp

    @property
    def voiced(self) -> bool:
        """Есть ли в кадре хотя бы одна звонкая половина."""
        return any(self.voicing)

    @property
    def rms(self) -> float:
        """RMS всего кадра."""
        return float(np.sqrt((self.gains[0] ** 2 + self.gains[1] ** 2) / 2))
