"""Скалярные квантователи с общим интерфейсом.

Любой квантователь переводит число в индекс из ``bits`` бит и обратно. Чтобы
попробовать другой способ квантования (например, адаптивный), достаточно
реализовать протокол :class:`Quantizer` и подставить его в режим кодека.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Quantizer(Protocol):
    bits: int

    def encode(self, x: float) -> int:
        """Значение -> индекс из bits бит."""
        ...

    def decode(self, index: int) -> float:
        """Индекс -> восстановленное значение."""
        ...


@dataclass
class UniformQuantizer:
    """Равномерный квантователь на отрезке [lo, hi] с восстановлением в центр ячейки."""

    lo: float
    hi: float
    bits: int

    @property
    def step(self) -> float:
        """Шаг квантования."""
        return (self.hi - self.lo) / (1 << self.bits)

    def encode(self, x: float) -> int:
        """Номер ячейки, в которую попало x (с ограничением по краям)."""
        return int(np.clip(np.floor((x - self.lo) / self.step), 0, (1 << self.bits) - 1))

    def decode(self, index: int) -> float:
        """Центр ячейки с номером index."""
        return self.lo + (index + 0.5) * self.step


@dataclass
class LarQuantizer:
    """Коэффициент отражения, равномерно квантованный в области LAR = log((1+k)/(1-k))."""

    lo: float
    hi: float
    bits: int

    def __post_init__(self):
        """Создаёт равномерный квантователь в области LAR."""
        self._q = UniformQuantizer(self.lo, self.hi, self.bits)

    def encode(self, k: float) -> int:
        """k -> LAR -> индекс."""
        k = np.clip(k, -0.9999, 0.9999)
        return self._q.encode(2.0 * np.arctanh(k))

    def decode(self, index: int) -> float:
        """Индекс -> LAR -> k (обратное преобразование через tanh)."""
        return float(np.tanh(self._q.decode(index) / 2.0))


@dataclass
class LogQuantizer:
    """Положительная величина, равномерно квантованная в децибелах.

    При silence_code=True индекс 0 означает "тишину" (всё, что тише lo_db), а остальные
    2^bits - 1 уровней равномерно покрывают [lo_db, hi_db]. Без этого всё тише lo_db
    поднималось бы до нижнего уровня - например, фон в паузах.
    """

    lo_db: float
    hi_db: float
    bits: int
    silence_code: bool = False
    silence_db: float = -100.0  # уровень, в который декодируется "тишина"

    def __post_init__(self):
        """Создаёт равномерный квантователь в децибелах."""
        if self.silence_code:
            # 2^bits - 1 уровней на [lo_db, hi_db]: квантователь на bits бит с тем же шагом,
            # у которого самый верхний уровень не используется
            step = (self.hi_db - self.lo_db) / ((1 << self.bits) - 1)
            self._q = UniformQuantizer(self.lo_db, self.hi_db + step, self.bits)
        else:
            self._q = UniformQuantizer(self.lo_db, self.hi_db, self.bits)

    def encode(self, x: float) -> int:
        """Амплитуда -> дБ -> индекс."""
        db = 20.0 * np.log10(max(x, 1e-12))
        if self.silence_code:
            if db < self.lo_db:
                return 0
            return min(self._q.encode(db) + 1, (1 << self.bits) - 1)
        return self._q.encode(db)

    def decode(self, index: int) -> float:
        """Индекс -> дБ -> амплитуда."""
        if self.silence_code:
            if index == 0:
                return float(10.0 ** (self.silence_db / 20.0))
            index -= 1
        return float(10.0 ** (self._q.decode(index) / 20.0))


class TableQuantizer:
    """Квантование к ближайшему значению таблицы (по относительной ошибке)."""

    def __init__(self, values: np.ndarray, bits: int):
        """values - допустимые значения (положительные), bits - разрядность индекса."""
        assert len(values) <= 1 << bits
        self.values = np.asarray(values, dtype=float)
        self.bits = bits
        self._log = np.log(self.values)

    def encode(self, x: float) -> int:
        """Индекс ближайшего по отношению значения таблицы."""
        return int(np.argmin(np.abs(self._log - np.log(x))))

    def decode(self, index: int) -> float:
        """Значение таблицы по индексу."""
        return float(self.values[min(index, len(self.values) - 1)])
