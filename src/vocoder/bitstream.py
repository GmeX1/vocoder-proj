"""Упаковка полей произвольной разрядности в поток битов."""

import struct


class BitWriter:
    """Накапливает поля в одно большое целое, старшие биты - первыми."""

    def __init__(self):
        """Создаёт пустой поток."""
        self._value = 0
        self._len = 0

    def write(self, value: int, bits: int) -> None:
        """Дописывает value разрядностью bits."""
        assert 0 <= value < (1 << bits), (value, bits)
        self._value = (self._value << bits) | value
        self._len += bits

    def __len__(self) -> int:
        """Число записанных бит."""
        return self._len

    def to_int(self) -> int:
        """Поток как целое число."""
        return self._value

    def to_bytes(self) -> bytes:
        """Поток как байты, дополненный нулями до целого байта."""
        pad = (-self._len) % 8
        return (self._value << pad).to_bytes((self._len + pad) // 8, "big")


class BitReader:
    """Читает поля из целого числа длиной length бит, начиная со старших."""

    def __init__(self, value: int, length: int):
        """value - биты потока, length - их количество."""
        self._value = value
        self._left = length

    @classmethod
    def from_bytes(cls, data: bytes) -> "BitReader":
        """Создаёт читатель из байтов."""
        return cls(int.from_bytes(data, "big"), len(data) * 8)

    def read(self, bits: int) -> int:
        """Читает следующее поле разрядностью bits."""
        assert bits <= self._left, "конец потока"
        self._left -= bits
        return (self._value >> self._left) & ((1 << bits) - 1)

    def remaining(self) -> int:
        """Сколько бит осталось."""
        return self._left


# Формат файла: магия, номер режима, число кадров, затем кадры подряд без выравнивания.
MAGIC = b"VOC1"
_HEADER = struct.Struct(">4sBI")


def pack_frames(codes: list[int], frame_bits: int, mode_id: int) -> bytes:
    """Собирает файл: заголовок + коды кадров подряд."""
    w = BitWriter()
    for c in codes:
        w.write(c, frame_bits)
    return _HEADER.pack(MAGIC, mode_id, len(codes)) + w.to_bytes()


def read_header(data: bytes) -> tuple[int, int, BitReader]:
    """Возвращает (номер режима, число кадров, читатель битов кадров)."""
    magic, mode_id, n = _HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError("не файл вокодера")
    return mode_id, n, BitReader.from_bytes(data[_HEADER.size :])
