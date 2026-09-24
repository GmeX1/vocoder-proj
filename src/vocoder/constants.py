"""Общие параметры вокодера."""

SAMPLE_RATE = 8000
FRAME_LEN = 180  # 22.5 мс
LPC_ORDER = 10
PREEMPHASIS = 0.9375

# Диапазон периода основного тона в отсчётах (~ 51-400 Гц).
MIN_PERIOD = 20
MAX_PERIOD = 156
