"""StarVLA's native MessagePack array format, without model-side dependencies.

The ndarray keys follow deployment/model_server/tools/msgpack_numpy.py in
starVLA/starVLA (MIT). Object and complex arrays are deliberately unsupported.
"""

import msgpack
import numpy as np


def _encode(value):
    if isinstance(value, np.ndarray | np.generic):
        if value.dtype.kind in "VOc":
            raise ValueError(f"Unsupported wire dtype: {value.dtype}")
        if isinstance(value, np.ndarray):
            return {
                b"__ndarray__": True,
                b"data": value.tobytes(),
                b"dtype": value.dtype.str,
                b"shape": value.shape,
            }
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"Unsupported wire value: {type(value).__name__}")


def _decode(value):
    if b"__ndarray__" in value or b"__npgeneric__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind in "VOc":
            raise ValueError(f"Unsupported wire dtype: {dtype}")
        if b"__ndarray__" in value:
            return np.ndarray(buffer=value[b"data"], dtype=dtype, shape=value[b"shape"])
        return dtype.type(value[b"data"])
    return value


def pack(value):
    return msgpack.packb(value, default=_encode, use_bin_type=True)


def unpack(value):
    if not isinstance(value, bytes):
        raise ValueError("StarVLA requires binary MessagePack frames")
    result = msgpack.unpackb(value, object_hook=_decode, raw=False, strict_map_key=False)
    if not isinstance(result, dict):
        raise ValueError("StarVLA frames must contain a mapping")
    return result
