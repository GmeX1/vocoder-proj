from vocoder.bitstream import BitReader, BitWriter, pack_frames, read_header


def test_bits_roundtrip():
    """Поля разной разрядности записываются и читаются без искажений."""
    fields = [(1, 1), (37, 7), (0, 5), (31, 5), (2, 2)]
    w = BitWriter()
    for v, b in fields:
        w.write(v, b)
    assert len(w) == 20
    r = BitReader.from_bytes(w.to_bytes())
    assert [r.read(b) for _, b in fields] == [v for v, _ in fields]


def test_file_roundtrip():
    """Файл с заголовком и кодами кадров читается обратно."""
    codes = [0, (1 << 54) - 1, 12345678901]
    mode_id, n, r = read_header(pack_frames(codes, 54, mode_id=1))
    assert (mode_id, n) == (1, 3)
    assert [r.read(54) for _ in range(n)] == codes
