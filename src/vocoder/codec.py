"""Кодер и декодер: связывают анализ, квантование и синтез."""

import numpy as np

from vocoder.analysis.analyzer import Analyzer
from vocoder.analysis.pitch import PitchDetector
from vocoder.bitstream import pack_frames, read_header
from vocoder.constants import SAMPLE_RATE
from vocoder.quant.lpc10 import Lpc10Mode
from vocoder.synthesis.synthesizer import Synthesizer

MODES = {m.mode_id: m for m in (Lpc10Mode,)}


def bitrate(mode) -> float:
    """Битрейт режима в бит/с."""
    return mode.frame_bits * SAMPLE_RATE / mode.frame_len


class Encoder:
    """Потоковый кодер: отсчёты -> коды кадров (целые числа по ``mode.frame_bits`` бит)."""

    def __init__(self, mode=None, pitch_detector: PitchDetector | None = None):
        """mode - режим квантования (по умолчанию LPC-10 2400), pitch_detector - определитель тона."""
        self.mode = mode or Lpc10Mode()
        self.analyzer = Analyzer(self.mode.frame_len, pitch_detector)
        self._n = 0

    def push(self, samples: np.ndarray) -> list[int]:
        """Добавляет отсчёты, возвращает коды готовых кадров."""
        return self._pack(self.analyzer.push(samples))

    def flush(self) -> list[int]:
        """Кодирует остаток сигнала в конце записи."""
        return self._pack(self.analyzer.flush())

    def _pack(self, frames) -> list[int]:
        """Упаковывает кадры, чередуя синхробит."""
        codes = []
        for f in frames:
            codes.append(self.mode.pack(f, sync=self._n))
            self._n += 1
        return codes


class Decoder:
    """Потоковый декодер: код кадра -> mode.frame_len отсчётов."""

    def __init__(self, mode=None):
        """mode - режим квантования, должен совпадать с режимом кодера."""
        self.mode = mode or Lpc10Mode()
        self.synth = Synthesizer(self.mode.frame_len)

    @property
    def delay(self) -> int:
        """Задержка выхода декодера в отсчётах."""
        return self.synth.delay

    def decode(self, code: int) -> np.ndarray:
        """Код кадра -> mode.frame_len отсчётов речи."""
        return self.synth.synthesize(self.mode.unpack(code))

    def flush(self) -> np.ndarray:
        """Выдаёт задержанный хвост в конце записи."""
        return self.synth.flush()


def encode_signal(x: np.ndarray, mode=None) -> bytes:
    """Кодирует весь сигнал целиком в байты файла."""
    enc = Encoder(mode)
    codes = enc.push(x) + enc.flush()
    return pack_frames(codes, enc.mode.frame_bits, enc.mode.mode_id)


def decode_bytes(data: bytes) -> np.ndarray:
    """Декодирует файл целиком; выход выровнен по времени со входом."""
    mode_id, n, reader = read_header(data)
    mode = MODES[mode_id]()
    dec = Decoder(mode)
    y = np.concatenate([dec.decode(reader.read(mode.frame_bits)) for _ in range(n)] + [dec.flush()])
    return y[dec.delay :]
