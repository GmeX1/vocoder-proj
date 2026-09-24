"""Командная строка: vocoder encode | decode | roundtrip."""

import argparse
from pathlib import Path

from vocoder.audio import load_audio, save_audio
from vocoder.codec import bitrate, decode_bytes, encode_signal
from vocoder.constants import SAMPLE_RATE
from vocoder.quant.lpc10 import Lpc10Mode


def main():
    """Разбирает аргументы и выполняет команду encode / decode / roundtrip."""
    ap = argparse.ArgumentParser(prog="vocoder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in [
        ("encode", "wav -> файл кодов"),
        ("decode", "файл кодов -> wav"),
        ("roundtrip", "wav -> коды -> wav"),
    ]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("input", type=Path)
        p.add_argument("output", type=Path)
    args = ap.parse_args()

    if args.cmd in ("encode", "roundtrip"):
        x = load_audio(args.input)
        data = encode_signal(x)
        seconds = len(x) / SAMPLE_RATE
        print(f"{seconds:.2f} с речи -> {len(data)} байт, режим {bitrate(Lpc10Mode()):.0f} бит/с")
        if args.cmd == "encode":
            args.output.write_bytes(data)
            return
        y = decode_bytes(data)[: len(x)]
    else:
        y = decode_bytes(args.input.read_bytes())
    save_audio(args.output, y)


if __name__ == "__main__":
    main()
