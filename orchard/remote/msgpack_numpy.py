"""
Copied from openpi project.

用法：
```py
import numpy as np

# 创建包含 NumPy 数据的数据结构
data = {
    "array_1d": np.array([1, 2, 3, 4, 5], dtype=np.float32),
    "array_2d": np.random.rand(3, 4),
    "scalar": np.int64(42),
    "regular_data": "normal string"
}

# 序列化
packed = packb(data)
print(f"序列化后大小: {len(packed)} bytes")

# 反序列化
unpacked = unpackb(packed)

print(f"原始数据: {data}")
print(f"原始类型: {type(data['array_1d'])}")
print(f"恢复后类型: {type(unpacked['array_1d'])}")
print(f"数据是否相等: {np.array_equal(data['array_1d'], unpacked['array_1d'])}")
```
"""

import functools

import msgpack
import numpy as np


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in (
        "V",
        "O",
        "c",
    ):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"]
        )

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
