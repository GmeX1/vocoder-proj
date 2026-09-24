"""Режим 2400 бит/с: распределение битов LPC-10 (FS-1015), 54 бита на кадр.

Таблица периодов и число бит на каждый параметр взяты из стандарта. Квантователи
коэффициентов - свои: равномерные, с диапазонами по статистике FLEURS-ru
(tools/param_stats.py, 1-й и 99-й процентили с запасом).

7-битный код тона и вокализованности (решение принимается для каждой половины кадра,
как в LPC-10; сама раскладка кодов - своя):
    0          обе половины глухие
    1..60      обе звонкие, полная таблица периодов
    61..90     глухая -> звонкая, каждый второй период таблицы
    91..120    звонкая -> глухая, каждый второй период таблицы

Отличие от стандарта: k1..k10 передаются во всех кадрах. В FS-1015 в глухих кадрах
передаются только k1..k4, а освободившиеся биты идут на защиту от ошибок; у нас защиты
нет, и полная огибающая заметно улучшает глухие звуки (STOI м 0.789 -> 0.812).
"""

import numpy as np

from vocoder.bitstream import BitReader, BitWriter
from vocoder.frame import FrameParams
from vocoder.quant.base import LarQuantizer, LogQuantizer, Quantizer, TableQuantizer, UniformQuantizer

# 60 периодов: 20-39 с шагом 1, 40-78 с шагом 2, 80-156 с шагом 4.
PITCH_TABLE = np.concatenate([np.arange(20, 40), np.arange(40, 80, 2), np.arange(80, 157, 4)])

COARSE_PITCH_TABLE = PITCH_TABLE[::2]  # 30 периодов для кадров с переходом

_UV_BASE = 1 + len(PITCH_TABLE)
_VU_BASE = _UV_BASE + len(COARSE_PITCH_TABLE)


class PitchVoicingCode:
    """Совместное кодирование периода тона и вокализованности двух половин кадра в 7 бит."""

    bits = 7

    def __init__(self):
        """Готовит таблицы периодов для полного и грубого квантования."""
        self.fine = TableQuantizer(PITCH_TABLE, bits=6)
        self.coarse = TableQuantizer(COARSE_PITCH_TABLE, bits=5)

    def encode(self, voicing: tuple[bool, bool], period: float) -> int:
        """(вокализованность половин, период) -> код 0..120."""
        match voicing:
            case (False, False):
                return 0
            case (True, True):
                return 1 + self.fine.encode(period)
            case (False, True):
                return _UV_BASE + self.coarse.encode(period)
            case _:
                return _VU_BASE + self.coarse.encode(period)

    def decode(self, code: int) -> tuple[tuple[bool, bool], float]:
        """Код -> (вокализованность половин, период). Неиспользуемые коды считаются глухими."""
        if 1 <= code < _UV_BASE:
            return (True, True), self.fine.decode(code - 1)
        if _UV_BASE <= code < _VU_BASE:
            return (False, True), self.coarse.decode(code - _UV_BASE)
        if _VU_BASE <= code < _VU_BASE + len(COARSE_PITCH_TABLE):
            return (True, False), self.coarse.decode(code - _VU_BASE)
        return (False, False), 0.0


class Lpc10Mode:
    """54 бита на кадр 22.5 мс: синхробит, тон+вокализованность, громкость, k1..k10."""

    mode_id = 1
    frame_len = 180
    frame_bits = 54

    def __init__(self):
        """Создаёт квантователи всех полей и проверяет, что их сумма - 54 бита."""
        self.pitch = PitchVoicingCode()
        # код 0 - тишина, 31 уровень на -66..-6 дБ (шаг ~ 1.9 дБ); запас сверху для громкого входа
        self.rms = LogQuantizer(-66.0, -6.0, bits=5, silence_code=True)
        self.rc: list[Quantizer] = [
            LarQuantizer(-4.6, 2.8, bits=5),
            LarQuantizer(-1.6, 3.8, bits=5),
            UniformQuantizer(-0.80, 0.70, bits=5),
            UniformQuantizer(-0.55, 0.90, bits=5),
            UniformQuantizer(-0.55, 0.70, bits=4),
            UniformQuantizer(-0.40, 0.80, bits=4),
            UniformQuantizer(-0.62, 0.62, bits=4),
            UniformQuantizer(-0.55, 0.67, bits=4),
            UniformQuantizer(-0.52, 0.50, bits=3),
            UniformQuantizer(-0.35, 0.50, bits=2),
        ]
        total = 1 + self.pitch.bits + self.rms.bits + sum(q.bits for q in self.rc)
        assert total == self.frame_bits, total

    def pack(self, p: FrameParams, sync: int) -> int:
        """Квантует параметры кадра и упаковывает их в 54-битное число."""
        w = BitWriter()
        w.write(sync & 1, 1)
        w.write(self.pitch.encode(p.voicing, p.period), self.pitch.bits)
        w.write(self.rms.encode(p.rms), self.rms.bits)
        for i, q in enumerate(self.rc):
            w.write(q.encode(p.rc[i]), q.bits)
        return w.to_int()

    def unpack(self, code: int) -> FrameParams:
        """Разбирает 54-битный код кадра обратно в параметры."""
        r = BitReader(code, self.frame_bits)
        r.read(1)  # синхробит
        voicing, period = self.pitch.decode(r.read(self.pitch.bits))
        rms = self.rms.decode(r.read(self.rms.bits))
        rc = np.array([q.decode(r.read(q.bits)) for q in self.rc])
        return FrameParams(voicing=voicing, period=period, gains=(rms, rms), rc=rc)
