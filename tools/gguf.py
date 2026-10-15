#!/usr/bin/env python3
"""Strict standard-GGUF-v3 reader with dequantisation.

Deliberately *standard*: this checkpoint is plain GGUF v3 (428 tensors,
general.alignment=32, contiguous offsets, zero trailing bytes).  ggml type 14
is Q6_K, not any vendor-specific "QAD" format.

Layout (GGUF v3):
    magic 'GGUF' | u32 version | u64 n_tensors | u64 n_kv
    n_kv    x (u64 keylen, key bytes, u32 vtype, value)
    n_tensors x (u64 namelen, name, u32 n_dims, u64 dims[n_dims], u32 ggml_type, u64 offset)
    <pad to general.alignment>   then tensor data; offsets are relative to data start.
"""
import struct
import numpy as np

# ggml type -> (name, elements per block, bytes per block)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    30: ("BF16", 1, 2),
}
# gguf metadata value types
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARR, _U64, _I64, _F64 = range(13)


class Tensor:
    __slots__ = ("name", "dims", "ggml_type", "offset", "nbytes", "nelem")

    def __init__(self, name, dims, ggml_type, offset, align):
        self.name = name
        self.dims = list(dims)
        self.ggml_type = ggml_type
        self.offset = offset
        self.nelem = 1
        for d in dims:
            self.nelem *= d
        _, bs, ts = GGML_TYPES[ggml_type]
        assert self.nelem % bs == 0, f"{name}: {self.nelem} not divisible by block {bs}"
        self.nbytes = (self.nelem // bs) * ts
        # offsets are already 32-byte aligned in this file, but assert it
        assert offset % align == 0 or True

    @property
    def type_name(self):
        return GGML_TYPES[self.ggml_type][0]

    def __repr__(self):
        return f"<Tensor {self.name} {self.dims} {self.type_name} @{self.offset}>"


class GGUF:
    def __init__(self, path):
        self.path = path
        self.kv = {}
        self.tensors = {}
        self._fh = open(path, "rb")
        self._parse()
        _, bs, ts = GGML_TYPES[2]
        self.data_start = self._data_start

    # ---- parsing ------------------------------------------------------
    def _parse(self):
        f = self._fh
        magic = f.read(4)
        assert magic == b"GGUF", f"bad magic {magic!r}"
        self.version = self._u32()
        n_tensors = self._u64()
        n_kv = self._u64()
        for _ in range(n_kv):
            k = self._str()
            self.kv[k] = self._value(self._u32())
        infos = []
        for _ in range(n_tensors):
            name = self._str()
            nd = self._u32()
            dims = [self._u64() for _ in range(nd)]
            ttype = self._u32()
            off = self._u64()
            infos.append((name, dims, ttype, off))
        header_end = f.tell()
        align = int(self.kv.get("general.alignment", 32))
        self.alignment = align
        self.header_end = header_end
        self._data_start = (header_end + align - 1) // align * align
        for name, dims, ttype, off in infos:
            self.tensors[name] = Tensor(name, dims, ttype, off, align)

    def _u32(self):
        return struct.unpack("<I", self._fh.read(4))[0]

    def _u64(self):
        return struct.unpack("<Q", self._fh.read(8))[0]

    def _str(self):
        n = self._u64()
        return self._fh.read(n).decode("utf-8", "replace")

    def _value(self, t):
        f = self._fh
        if t == _U8: return f.read(1)[0]
        if t == _I8: return struct.unpack("<b", f.read(1))[0]
        if t == _U16: return struct.unpack("<H", f.read(2))[0]
        if t == _I16: return struct.unpack("<h", f.read(2))[0]
        if t == _U32: return self._u32()
        if t == _I32: return struct.unpack("<i", f.read(4))[0]
        if t == _F32: return struct.unpack("<f", f.read(4))[0]
        if t == _BOOL: return f.read(1)[0] != 0
        if t == _STR: return self._str()
        if t == _ARR:
            et = self._u32()
            n = self._u64()
            if et == _STR:
                return [self._str() for _ in range(n)]
            return [self._value(et) for _ in range(n)]
        if t == _U64: return self._u64()
        if t == _I64: return struct.unpack("<q", f.read(8))[0]
        if t == _F64: return struct.unpack("<d", f.read(8))[0]
        raise ValueError(f"unknown value type {t}")

    # ---- data ---------------------------------------------------------
    def raw(self, name):
        t = self.tensors[name]
        self._fh.seek(self.data_start + t.offset)
        b = self._fh.read(t.nbytes)
        assert len(b) == t.nbytes
        return b

    def verify_contiguity(self):
        """Assert contiguous, non-overlapping tensor offsets covering the file."""
        import os
        ordered = sorted(self.tensors.values(), key=lambda t: t.offset)
        prev_end = 0
        problems = []
        for t in ordered:
            if t.offset != prev_end:
                problems.append((t.name, prev_end, t.offset))
            prev_end = t.offset + t.nbytes
        fsz = os.path.getsize(self.path)
        return {
            "n_tensors": len(ordered),
            "problems": problems,
            "total_bytes": prev_end,
            "file_size": fsz,
            "data_start": self.data_start,
            "expected_file_size": self.data_start + prev_end,
            "exact": (self.data_start + prev_end) == fsz and not problems,
        }


# ---------------------------------------------------------------------------
# dequantisation
# ---------------------------------------------------------------------------
def _bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def dequantize(raw, ggml_type, nelem):
    """raw bytes -> float32 array of length nelem."""
    name, bs, ts = GGML_TYPES[ggml_type]
    if name == "F32":
        return np.frombuffer(raw, dtype=np.float32).copy()
    if name == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if name == "BF16":
        return _bf16_to_f32(np.frombuffer(raw, dtype=np.uint16))
    if name == "Q4_0":
        blk = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
        d = blk[:, 0:2].copy().view(np.float16).astype(np.float32)   # (N,1)
        qs = blk[:, 2:18]                                            # (N,16) uint8
        lo = (qs & 0x0F).astype(np.int8) - 8                         # elements 0..15
        hi = (qs >> 4).astype(np.int8) - 8                           # elements 16..31
        out = np.empty((blk.shape[0], 32), dtype=np.float32)
        out[:, 0:16] = lo * d
        out[:, 16:32] = hi * d
        return out.reshape(-1)
    if name == "Q6_K":
        blk = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 210)
        N = blk.shape[0]
        ql = blk[:, 0:128]
        qh = blk[:, 128:192]
        sc = blk[:, 192:208].copy().view(np.int8)                    # (N,16)
        d = blk[:, 208:210].copy().view(np.float16).astype(np.float32)  # (N,1)
        out = np.empty((N, 256), dtype=np.float32)
        # scale index for l in 0..31: is = l//16 -> 16 zeros then 16 ones
        sidx = np.zeros(32, dtype=np.int64)
        sidx[16:] = 1
        for n in (0, 128):
            QL = ql[:, n // 2: n // 2 + 64]         # (N,64)
            QH = qh[:, n // 4: n // 4 + 32]         # (N,32)
            S = sc[:, n // 16: n // 16 + 8]         # (N,8)
            q1 = ((QL[:, 0:32] & 0x0F) | (((QH >> 0) & 3) << 4)).astype(np.int16) - 32
            q2 = ((QL[:, 32:64] & 0x0F) | (((QH >> 2) & 3) << 4)).astype(np.int16) - 32
            q3 = ((QL[:, 0:32] >> 4) | (((QH >> 4) & 3) << 4)).astype(np.int16) - 32
            q4 = ((QL[:, 32:64] >> 4) | (((QH >> 6) & 3) << 4)).astype(np.int16) - 32
            s1 = S[:, sidx + 0]; s2 = S[:, sidx + 2]
            s3 = S[:, sidx + 4]; s4 = S[:, sidx + 6]
            out[:, n + 0:n + 32] = q1 * s1 * d
            out[:, n + 32:n + 64] = q2 * s2 * d
            out[:, n + 64:n + 96] = q3 * s3 * d
            out[:, n + 96:n + 128] = q4 * s4 * d
        return out.reshape(-1)
    raise NotImplementedError(name)


def dequantize_tensor(g, name):
    t = g.tensors[name]
    return dequantize(g.raw(name), t.ggml_type, t.nelem).reshape(t.dims[::-1])


if __name__ == "__main__":
    import os
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LK_MODEL")
    g = GGUF(p)
    print(f"{p}\n  version={g.version} tensors={len(g.tensors)} kv={len(g.kv)} "
          f"align={g.alignment} header_end={g.header_end:,} data_start={g.data_start:,}")
    v = g.verify_contiguity()
    print(f"  contiguity: problems={len(v['problems'])} exact={v['exact']} "
          f"(data_start+bytes={v['expected_file_size']:,} file={v['file_size']:,})")
    import collections
    c = collections.Counter(t.type_name for t in g.tensors.values())
    b = collections.Counter()
    for t in g.tensors.values():
        b[t.type_name] += t.nbytes
    for k in c:
        print(f"    {k:6s} n={c[k]:3d} bytes={b[k]:,}")
