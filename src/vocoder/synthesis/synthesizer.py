"""Синтез речи из параметров кадров."""

import numpy as np
from scipy.signal import lfilter, lfiltic

from vocoder.analysis.lpc import prediction_gain, rc_to_lpc
from vocoder.constants import FRAME_LEN, LPC_ORDER, PREEMPHASIS
from vocoder.frame import FrameParams

SUBBLOCKS = 4  # параметры интерполируются внутри кадра по 4 подблокам


def soft_clip(x: np.ndarray, knee: float = 0.8) -> np.ndarray:
    """Мягкое ограничение: до knee сигнал не меняется, выше плавно подходит к +-1 без щелчков."""
    a = np.abs(x)
    over = a > knee
    y = x.copy()
    y[over] = np.sign(x[over]) * (knee + (1 - knee) * np.tanh((a[over] - knee) / (1 - knee)))
    return y


class Synthesizer:
    """Потоковый синтезатор: параметры кадра -> frame_len отсчётов.

    Кадр n синтезируется на отрезке между центрами кадров n-1 и n, то есть это
    вторая половина кадра n-1 и первая половина кадра n. Поэтому выход задержан
    относительно входа на ``delay`` отсчётов.
    """

    def __init__(self, frame_len: int = FRAME_LEN, seed: int = 0):
        """frame_len - длина кадра; seed делает шумовое возбуждение воспроизводимым."""
        self.frame_len = frame_len
        self.delay = frame_len // 2
        self._rng = np.random.default_rng(seed)
        self._prev: FrameParams | None = None
        self._y_hist = np.zeros(LPC_ORDER)  # последние выходы фильтра синтеза, по времени
        self._deemph_zi = np.zeros(1)
        self._next_pulse = 0.0  # позиция следующего импульса относительно начала подблока
        self._scale = 1.0  # поправка громкости, применённая в конце прошлого кадра

    def synthesize(self, cur: FrameParams) -> np.ndarray:
        """Синтезирует frame_len отсчётов, плавно переходя от параметров прошлого кадра к cur."""
        prev = self._prev or cur
        sub = self.frame_len // SUBBLOCKS
        gains_db = self._gain_track(prev, cur)
        out = np.empty(self.frame_len)
        for j in range(SUBBLOCKS):
            w = (j + 0.5) / SUBBLOCKS
            rc = (1 - w) * prev.rc + w * cur.rc
            voiced = prev.voicing[1] if j < SUBBLOCKS // 2 else cur.voicing[0]
            period = self._period(prev, cur, w)

            sigma = 10 ** (gains_db[j] / 20) * np.sqrt(prediction_gain(rc))  # оценка по теории, уточняется ниже
            exc = self._pulses(sub, period, sigma) if voiced and period > 0 else self._noise(sub, sigma)

            a = rc_to_lpc(rc)
            zi = lfiltic([1.0], a, self._y_hist[::-1])
            y, _ = lfilter([1.0], a, exc, zi=zi)
            self._y_hist = np.concatenate([self._y_hist, y])[-LPC_ORDER:]
            out[j * sub : (j + 1) * sub] = y

        # Формула для sigma точна только для шумового возбуждения; для импульсов энергия на
        # выходе фильтра зависит от того, как гармоники тона ложатся на форманты. Поэтому
        # громкость кадра подгоняется по фактической энергии (как в LPC-10e). Поправка
        # меняется плавно от значения прошлого кадра, чтобы не было скачков на границах.
        target = sub * np.sum(10 ** (gains_db / 10))
        scale = float(np.clip(np.sqrt(target / (np.sum(out**2) + 1e-12)), 0.25, 4.0))
        out *= np.linspace(self._scale, scale, self.frame_len + 1)[1:]
        self._y_hist *= scale
        self._scale = scale

        self._prev = cur
        out, self._deemph_zi = lfilter([1.0], [1.0, -PREEMPHASIS], out, zi=self._deemph_zi)
        return soft_clip(out)

    def flush(self) -> np.ndarray:
        """Досинтезирует задержанный хвост сигнала, повторяя параметры последнего кадра."""
        if self._prev is None:
            return np.zeros(0)
        return self.synthesize(self._prev)[: self.delay]

    def _gain_track(self, prev: FrameParams, cur: FrameParams) -> np.ndarray:
        """Громкость (дБ) в центрах подблоков: линейная интерполяция между центрами половин кадров.

        Отсчёт от центра кадра n-1: центры половин - -L/4, +L/4 (кадр n-1), 3L/4, 5L/4 (кадр n).
        """
        L = self.frame_len
        anchors_t = np.array([-L / 4, L / 4, 3 * L / 4, 5 * L / 4])
        anchors_db = 20 * np.log10(np.array([*prev.gains, *cur.gains]) + 1e-9)
        centers = (np.arange(SUBBLOCKS) + 0.5) * L / SUBBLOCKS
        return np.interp(centers, anchors_t, anchors_db)

    @staticmethod
    def _period(prev: FrameParams, cur: FrameParams, w: float) -> float:
        """Период для подблока с весом w текущего кадра: интерполяция, если тон есть в обоих кадрах."""
        if prev.period and cur.period:
            return (1 - w) * prev.period + w * cur.period
        return cur.period or prev.period

    def _pulses(self, n: int, period: float, sigma: float) -> np.ndarray:
        """Последовательность импульсов со средней мощностью sigma^2."""
        exc = np.zeros(n)
        amp = sigma * np.sqrt(period)
        pos = self._next_pulse
        while pos < n:
            exc[int(pos)] = amp
            pos += period
        self._next_pulse = pos - n
        return exc

    def _noise(self, n: int, sigma: float) -> np.ndarray:
        """Белый шум со среднеквадратичным значением sigma; сбрасывает фазу импульсов."""
        self._next_pulse = 0.0
        return sigma * self._rng.standard_normal(n)
