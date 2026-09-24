"""Низкоскоростной LPC-вокодер."""

from vocoder.codec import Decoder, Encoder, decode_bytes, encode_signal

__all__ = ["Encoder", "Decoder", "encode_signal", "decode_bytes"]
