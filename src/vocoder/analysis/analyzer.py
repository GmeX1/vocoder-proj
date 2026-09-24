"""Потоковый анализатор: отсчёты речи -> параметры кадров."""

import numpy as np
from scipy.signal import butter, lfilter, sosfilt, sosfilt_zi

from vocoder.analysis.lpc import autocorrelation, levinson_durbin
from vocoder.analysis.pitch import PitchDetector, PitchEstimate, YinPitchDetector
from vocoder.analysis.voicing import VoicingDetector, VoicingFeatures, voicing_features
from vocoder.constants import FRAME_LEN, LPC_ORDER, PREEMPHASIS, SAMPLE_RATE
from vocoder.frame import FrameParams

PITCH_LOWPASS_HZ = 900.0


def track_octave(est: PitchEstimate, ref: float, near: float = 0.12, far: float = 0.25) -> PitchEstimate:
    """Исправляет кратные ошибки тона, выбирая кандидата YIN рядом с ожидаемым периодом ref.

    Замена делается, только если лучшая оценка отличается от ref больше чем на far, а рядом
    с ref (в пределах near) есть провал d'(tau). Переход к более короткому периоду допускается
    при провале не намного мельче лучшего; к более длинному - только если провал заметно
    глубже: провал на кратном периоде есть у любого периодического сигнала.
    """
    if abs(est.period / ref - 1) <= far:
        return est
    for period, value in est.candidates:
        if abs(period / ref - 1) > near:
            continue
        shorter = period < est.period
        if (shorter and value <= est.aperiodicity + 0.10) or (not shorter and value < est.aperiodicity - 0.05):
            return PitchEstimate(period, value, est.candidates)
    return est


class Analyzer:
    """Принимает сигнал произвольными порциями и выдаёт параметры готовых кадров.

    Окна анализа центрированы на кадре (LPC) и на половинах кадра (тон) и заходят
    за границы кадра. Кроме того, решению "звонкий/глухой" нужны признаки двух
    следующих полукадров. Поэтому кадр выдаётся с задержкой ``lookahead`` отсчётов
    после своего конца.
    """

    def __init__(
        self,
        frame_len: int = FRAME_LEN,
        pitch_detector: PitchDetector | None = None,
        voicing: VoicingDetector | None = None,
    ):
        """frame_len - длина кадра; детекторы тона и вокализованности можно подменить."""
        self.frame_len = frame_len
        self.pitch = pitch_detector or YinPitchDetector()
        self.voicing = voicing or VoicingDetector()
        self.lpc_window = np.hamming(frame_len * 4 // 3)  # окно на треть длиннее кадра

        quarter = frame_len // 4  # смещение центра половины кадра от центра кадра
        self._half = max(len(self.lpc_window) // 2, self.pitch.window_len // 2 + quarter) + 1
        self.lookahead = self._half - frame_len // 2 + self.voicing.DELAY * frame_len // 2

        self._pre_zi = np.zeros(1)
        self._lp_sos = butter(4, PITCH_LOWPASS_HZ, fs=SAMPLE_RATE, output="sos")
        self._lp_zi = sosfilt_zi(self._lp_sos) * 0.0

        # Буферы начинаются с тишины, чтобы окна первого кадра могли заглянуть "в прошлое".
        history = self._half - frame_len // 2
        self._raw = np.zeros(history)
        self._pre = np.zeros(history)
        self._low = np.zeros(history)
        self._total = 0  # всего поступило отсчётов
        self.feature_log: list[VoicingFeatures] | None = None  # если список - сюда пишутся признаки половин кадров
        self._pending: list[dict] = []  # кадры, ждущие решений "звонкий/глухой"
        self._decisions: list[bool] = []  # готовые решения для полукадров из _pending
        self._last_period = 0.0  # период последнего выданного звонкого кадра
        self._since_voiced = 99  # сколько кадров назад он был

    def push(self, samples: np.ndarray) -> list[FrameParams]:
        """Добавляет порцию отсчётов и возвращает параметры кадров, ставших готовыми."""
        samples = np.asarray(samples, dtype=float)
        pre, self._pre_zi = lfilter([1.0, -PREEMPHASIS], [1.0], samples, zi=self._pre_zi)
        low, self._lp_zi = sosfilt(self._lp_sos, samples, zi=self._lp_zi)
        self._raw = np.concatenate([self._raw, samples])
        self._pre = np.concatenate([self._pre, pre])
        self._low = np.concatenate([self._low, low])
        self._total += len(samples)

        # центр очередного кадра всегда находится в буфере по индексу self._half
        while len(self._raw) >= 2 * self._half:
            self._analyze(self._half)
            self._raw = self._raw[self.frame_len :]
            self._pre = self._pre[self.frame_len :]
            self._low = self._low[self.frame_len :]
        return self._ready()

    def flush(self) -> list[FrameParams]:
        """Дополняет сигнал тишиной до конца последнего кадра и выдаёт все оставшиеся кадры."""
        pad = (-self._total) % self.frame_len + self._half - self.frame_len // 2
        frames = self.push(np.zeros(pad))
        self._decisions += self.voicing.flush()
        self._total = 0
        return frames + self._ready()

    def _analyze(self, c: int) -> None:
        """Считает параметры кадра с центром в индексе c буфера и ставит его в очередь."""
        h = self.frame_len // 2
        rc = self._lpc(c)
        estimates, gains = [], []
        for start in (c - h, c):  # первая и вторая половины кадра
            est = self._pitch_at(start + h // 2)
            feats = voicing_features(
                self._raw[start : start + h], self._low[start : start + h], est.aperiodicity, rc[0], est.period
            )
            if self.feature_log is not None:
                self.feature_log.append(feats)
            self._decisions += self.voicing.push(feats)
            estimates.append(est)
            gains.append(float(np.sqrt(np.mean(self._pre[start : start + h] ** 2))))
        self._pending.append({"rc": rc, "estimates": estimates, "gains": gains})

    def _ready(self) -> list[FrameParams]:
        """Выдаёт кадры, для обеих половин которых уже есть решения "звонкий/глухой"."""
        frames = []
        while self._pending and len(self._decisions) >= 2:
            fr = self._pending.pop(0)
            voicing = [self._decisions.pop(0), self._decisions.pop(0)]
            ref = self._reference_period()
            estimates = [track_octave(e, ref) if v and ref else e for v, e in zip(voicing, fr["estimates"])]
            period = self._frame_period(voicing, estimates)
            if period:
                self._last_period, self._since_voiced = period, 0
            else:
                self._since_voiced += 1
            frames.append(
                FrameParams(
                    voicing=(voicing[0], voicing[1]),
                    period=period,
                    gains=(fr["gains"][0], fr["gains"][1]),
                    rc=fr["rc"],
                )
            )
        return frames

    def _reference_period(self) -> float:
        """Ожидаемый период: тон предыдущего кадра, если он был недавно, иначе надёжная оценка следующего."""
        if self._since_voiced <= 1:
            return self._last_period
        if self._pending:
            best = min(self._pending[0]["estimates"], key=lambda e: e.aperiodicity)
            if best.aperiodicity < 0.15:
                return best.period
        return 0.0

    def _lpc(self, c: int) -> np.ndarray:
        """Коэффициенты отражения по окну Хэмминга с центром в c."""
        lw = len(self.lpc_window) // 2
        r = autocorrelation(self._pre[c - lw : c + lw] * self.lpc_window, LPC_ORDER)
        if r[0] <= 1e-10:
            return np.zeros(LPC_ORDER)
        r[0] *= 1.0001  # "шумовой пол" -40 дБ: защита от плохой обусловленности
        return levinson_durbin(r, LPC_ORDER)[1]

    def _pitch_at(self, center: int) -> PitchEstimate:
        """Оценка тона по окну детектора с центром в индексе center."""
        pw = self.pitch.window_len
        return self.pitch.estimate(self._low[center - pw // 2 : center - pw // 2 + pw])

    @staticmethod
    def _frame_period(voicing: list[bool], est: list[PitchEstimate]) -> float:
        """Один период на кадр: среднее по звонким половинам, при расхождении - более надёжная."""
        voiced = [e for v, e in zip(voicing, est) if v]
        if not voiced:
            return 0.0
        if len(voiced) == 2 and abs(voiced[0].period / voiced[1].period - 1) < 0.15:
            return (voiced[0].period + voiced[1].period) / 2
        return min(voiced, key=lambda e: e.aperiodicity).period
