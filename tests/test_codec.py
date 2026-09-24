import numpy as np
import pytest

from vocoder.codec import Decoder, Encoder, bitrate, decode_bytes, encode_signal
from vocoder.constants import FRAME_LEN
from vocoder.frame import FrameParams
from vocoder.quant.lpc10 import Lpc10Mode


def _speechlike(n, seed=0):
    """Половина сигнала - импульсы (звонкая), половина - шум (глухая)."""
    rng = np.random.default_rng(seed)
    exc = np.zeros(n)
    exc[::60] = 1.0
    exc[n // 2 :] = 0.3 * rng.standard_normal(n - n // 2)  # звонкая половина + шумовая
    from scipy.signal import lfilter

    return 0.05 * lfilter([1.0], [1.0, -1.3, 0.8], exc)


def test_bitrate_2400():
    """54 бита на 22.5 мс дают ровно 2400 бит/с."""
    assert bitrate(Lpc10Mode()) == pytest.approx(2400)


def test_frame_count_and_length():
    """Выход не короче входа и длиннее не больше чем на кадр."""
    x = _speechlike(8000 + 77)
    y = decode_bytes(encode_signal(x))
    assert len(y) >= len(x)
    assert len(y) - len(x) < FRAME_LEN


def test_streaming_equals_offline():
    """Кодирование порциями даёт те же коды, что и целиком."""
    x = _speechlike(6000)
    enc = Encoder()
    codes = []
    for chunk in np.array_split(x, 37):
        codes += enc.push(chunk)
    codes += enc.flush()
    offline = Encoder()
    assert codes == offline.push(x) + offline.flush()


def test_pack_unpack_voiced_frame():
    """Звонкий кадр проходит упаковку с погрешностью в пределах шага квантования."""
    mode = Lpc10Mode()
    rc = np.array([-0.9, 0.4, -0.2, 0.3, 0.1, 0.2, 0.0, 0.1, 0.0, 0.1])
    p = FrameParams(voicing=(True, True), period=100.0, gains=(0.01, 0.01), rc=rc)
    q = mode.unpack(mode.pack(p, sync=1))
    assert q.voicing == (True, True) and q.period == 100
    assert q.rms == pytest.approx(0.01, rel=0.15)
    np.testing.assert_allclose(q.rc, rc, atol=0.15)


def test_unvoiced_frame_keeps_all_rc():
    """В глухом кадре передаются все 10 коэффициентов (без защиты от ошибок они нужнее)."""
    mode = Lpc10Mode()
    p = FrameParams(voicing=(False, False), period=0.0, gains=(0.01, 0.01), rc=np.full(10, 0.3))
    q = mode.unpack(mode.pack(p, sync=0))
    assert q.voicing == (False, False) and q.period == 0
    np.testing.assert_allclose(q.rc, 0.3, atol=0.15)


def test_decoder_output_is_finite():
    """Декодер не выдаёт NaN/inf и не выходит за [-1, 1]."""
    x = _speechlike(4000)
    enc, dec = Encoder(), Decoder()
    y = np.concatenate([dec.decode(c) for c in enc.push(x) + enc.flush()])
    assert np.all(np.isfinite(y)) and np.abs(y).max() <= 1.0


@pytest.mark.parametrize("voicing", [(False, True), (True, False)])
def test_voicing_transitions_roundtrip(voicing):
    """Кадры с переходом звонкий/глухой сохраняют решения половин и грубый период."""
    mode = Lpc10Mode()
    p = FrameParams(voicing=voicing, period=90.0, gains=(0.01, 0.02), rc=np.zeros(10))
    q = mode.unpack(mode.pack(p, sync=0))
    assert q.voicing == voicing
    assert q.period == pytest.approx(90.0, rel=0.06)


def test_rms_silence_code():
    """Громкость тише нижнего уровня кодируется как тишина, а не поднимается до него."""
    q = Lpc10Mode().rms
    assert q.encode(1e-6) == 0 and q.decode(0) < 1e-4
    top = q.encode(10 ** (-6 / 20))
    assert top == 31 and abs(20 * np.log10(q.decode(top)) + 6) < 1.0
    mid = 10 ** (-30 / 20)
    assert abs(20 * np.log10(q.decode(q.encode(mid))) + 30) < 1.0
