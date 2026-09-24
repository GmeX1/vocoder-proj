"""Решение "звонкий / глухой" для половин кадров."""

from dataclasses import dataclass

import numpy as np


@dataclass
class VoicingFeatures:
    """Признаки половины кадра, которые считает анализатор."""

    level_db: float  # уровень, dBFS
    aperiodicity: float  # минимум d'(tau) из YIN: 0 - периодичный сигнал, ~1 - шум
    zcr: float  # доля пересечений нуля между соседними отсчётами
    low_ratio: float  # доля энергии ниже 900 Гц
    k1: float  # первый коэффициент отражения (наклон спектра)
    period: float = 0.0  # период тона, найденный YIN


def voicing_features(
    raw: np.ndarray, low: np.ndarray, aperiodicity: float, k1: float, period: float = 0.0
) -> VoicingFeatures:
    """Считает признаки по отрезку сигнала raw и его низкочастотной версии low."""
    energy = np.sum(raw**2) + 1e-12
    return VoicingFeatures(
        level_db=float(10 * np.log10(energy / len(raw))),
        aperiodicity=float(aperiodicity),
        zcr=float(np.mean(np.abs(np.diff(np.signbit(raw).astype(np.int8))))),
        low_ratio=float(min(np.sum(low**2) / energy, 1.0)),
        k1=float(k1),
        period=float(period),
    )


class LevelTracker:
    """Уровень относительно недавнего пика речи: пик мгновенно растёт и медленно спадает."""

    def __init__(self, decay_db: float = 0.05, floor_db: float = -45.0, min_rel_db: float = -40.0):
        """decay_db - спад пика за полукадр; floor_db - пик не опускается ниже; min_rel_db - нижняя граница результата."""
        self.decay_db = decay_db
        self.floor_db = floor_db
        self.min_rel_db = min_rel_db
        self._peak: float | None = None

    def relative(self, level_db: float) -> float:
        """Обновляет пик и возвращает level_db - пик (<= 0)."""
        self._peak = level_db if self._peak is None else max(level_db, self._peak - self.decay_db)
        self._peak = max(self._peak, self.floor_db)
        return max(level_db - self._peak, self.min_rel_db)


class VoicingDetector:
    """Потоковый классификатор "звонкий / глухой" с контекстом соседних полукадров.

    Для полукадра i считается вероятность по логистической регрессии от признаков:
    апериодичность, пересечения нуля, доля низких частот, k1, относительный уровень,
    апериодичность полукадров i-1 и i+1. Затем вероятность сглаживается медианой по
    полукадрам i-1, i, i+1. Поэтому решение для полукадра i готово только после
    прихода признаков полукадра i+2 (задержка DELAY полукадров).

    Веса обучены на 200 записях FLEURS-ru (не из тестового набора) с разметкой Praat
    (tools/train_voicing.py).
    """

    DELAY = 2
    # Порядок: апериодичность, пересечения нуля, доля низких частот, k1, отн. уровень / 10,
    # апериодичность предыдущего и следующего полукадров.
    WEIGHTS = np.array([-3.966, -0.162, 1.192, -1.067, 1.511, -2.649, -4.232])
    BIAS = 4.585

    def __init__(self, weights: np.ndarray | None = None, bias: float | None = None, silence_db: float = -60.0):
        """weights/bias - параметры классификатора; полукадры тише silence_db всегда глухие."""
        self.weights = self.WEIGHTS if weights is None else np.asarray(weights)
        self.bias = self.BIAS if bias is None else bias
        self.silence_db = silence_db
        self._level = LevelTracker()
        self._feats: list[tuple[VoicingFeatures, float]] = []  # (признаки, отн. уровень) ещё не решённых полукадров
        self._probs: list[float] = []  # вероятности для тех же полукадров, по мере готовности
        self._prev_ap: float | None = None  # апериодичность полукадра перед первым в _feats
        self._prev_prob: float | None = None  # вероятность полукадра перед первым в _feats

    @staticmethod
    def vector(f: VoicingFeatures, rel_db: float, prev_ap: float, next_ap: float) -> np.ndarray:
        """Вектор признаков для классификатора."""
        return np.array([f.aperiodicity, f.zcr, f.low_ratio, f.k1, rel_db / 10, prev_ap, next_ap])

    def probability(self, v: np.ndarray) -> float:
        """Вероятность того, что полукадр звонкий."""
        return float(1.0 / (1.0 + np.exp(-(self.weights @ v + self.bias))))

    def push(self, f: VoicingFeatures) -> list[bool]:
        """Принимает признаки очередного полукадра, возвращает решения, ставшие окончательными."""
        self._feats.append((f, self._level.relative(f.level_db)))
        return self._advance(final=False)

    def flush(self) -> list[bool]:
        """Выдаёт решения для оставшихся полукадров в конце записи."""
        return self._advance(final=True)

    def _advance(self, final: bool) -> list[bool]:
        """Досчитывает готовые вероятности и решения."""
        # вероятность полукадра j требует признаков j+1 (или конца записи)
        while len(self._probs) < len(self._feats) - (0 if final else 1):
            j = len(self._probs)
            f, rel = self._feats[j]
            if j > 0:
                prev_ap = self._feats[j - 1][0].aperiodicity
            else:
                prev_ap = f.aperiodicity if self._prev_ap is None else self._prev_ap
            nxt = self._feats[j + 1][0].aperiodicity if j + 1 < len(self._feats) else f.aperiodicity
            self._probs.append(self.probability(self.vector(f, rel, prev_ap, nxt)))

        # решение для первого нерешённого полукадра требует вероятности следующего (или конца записи)
        out = []
        while self._probs and (len(self._probs) >= 2 or final):
            p = self._probs[0]
            prev_p = p if self._prev_prob is None else self._prev_prob
            next_p = self._probs[1] if len(self._probs) > 1 else p
            f, _ = self._feats[0]
            out.append(bool(np.median([prev_p, p, next_p]) >= 0.5 and f.level_db >= self.silence_db))
            self._prev_prob, self._prev_ap = p, f.aperiodicity
            self._probs.pop(0)
            self._feats.pop(0)
        return out

    def decide_all(self, feats: list[VoicingFeatures]) -> list[bool]:
        """Решения для целой последовательности (офлайн); совпадает с потоковым режимом."""
        out = []
        for f in feats:
            out += self.push(f)
        return out + self.flush()
