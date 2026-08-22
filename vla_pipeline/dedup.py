"""
去重：网络视频最大的问题不是量少，是重复。

同一段素材会以不同分辨率、不同片头、不同压制参数出现在多个来源。
按 URL 或文件哈希去重完全挡不住这类近似重复——像素级差一点点，MD5 就完全不同。

做法：
  1. 用 ffmpeg 均匀抽 K 帧，直接输出 32x32 灰度裸数据（不解码成图片，省一个依赖）
  2. 每帧算 64bit pHash（DCT-II 取低频 8x8）
  3. 片段间比对：逐帧汉明距离取均值，低于阈值判为近似重复
  4. 并查集聚类，每簇保留分辨率最高的一条

依赖：ffmpeg 二进制 + numpy。不需要 opencv / imagehash / PIL。
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

HASH_SIZE = 8       # 最终 8x8 = 64 bit
DCT_SIZE = 32       # 先缩到 32x32 再取 DCT 低频
DEFAULT_FRAMES = 5  # 每段抽 5 帧

# uint8 popcount 查表，用于向量化汉明距离
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _dct_matrix(n: int) -> np.ndarray:
    """DCT-II 变换矩阵。用矩阵乘法代替 scipy.fft，少一个依赖。"""
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    m[0, :] /= np.sqrt(2)
    return m * np.sqrt(2.0 / n)


_DCT = _dct_matrix(DCT_SIZE)


def extract_gray_frames(
    path: str | Path,
    duration_ms: int,
    n_frames: int = DEFAULT_FRAMES,
) -> np.ndarray:
    """一次 ffmpeg 调用抽 n 帧 32x32 灰度，返回 (n, 32, 32) float32。"""
    duration_s = max(duration_ms / 1000.0, 0.1)
    rate = max(n_frames / duration_s, 0.05)  # 太小会让 ffmpeg 抽不出帧
    cmd = [
        FFMPEG, "-v", "error", "-i", str(path),
        "-vf", f"fps={rate:.6f},scale={DCT_SIZE}:{DCT_SIZE}",
        "-frames:v", str(n_frames),
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    r = subprocess.run(cmd, capture_output=True, check=False)
    frame_bytes = DCT_SIZE * DCT_SIZE
    n_got = len(r.stdout) // frame_bytes
    if n_got == 0:
        return np.zeros((0, DCT_SIZE, DCT_SIZE), dtype=np.float32)
    buf = np.frombuffer(r.stdout[: n_got * frame_bytes], dtype=np.uint8)
    return buf.reshape(n_got, DCT_SIZE, DCT_SIZE).astype(np.float32)


def phash_frame(gray: np.ndarray) -> np.uint64:
    """标准 pHash：DCT 后取左上 8x8（去掉 DC），跟中位数比较得到 64bit。"""
    coeffs = _DCT @ gray @ _DCT.T
    block = coeffs[:HASH_SIZE, :HASH_SIZE].flatten()
    med = np.median(block[1:])          # 排除 DC 分量，它只反映整体亮度
    bits = (block > med).astype(np.uint64)
    out = np.uint64(0)
    for b in bits:
        out = np.uint64(out << np.uint64(1)) | np.uint64(b)
    return out


def clip_signature(
    path: str | Path, duration_ms: int, n_frames: int = DEFAULT_FRAMES
) -> np.ndarray:
    """片段签名 = n 个 64bit pHash。帧数不足时用最后一帧补齐，保证长度一致。"""
    frames = extract_gray_frames(path, duration_ms, n_frames)
    if len(frames) == 0:
        return np.zeros(n_frames, dtype=np.uint64)
    hashes = [phash_frame(f) for f in frames]
    while len(hashes) < n_frames:
        hashes.append(hashes[-1])
    return np.array(hashes[:n_frames], dtype=np.uint64)


def hamming_matrix(sigs: np.ndarray) -> np.ndarray:
    """
    (N, K) uint64 签名 -> (N, N) 平均汉明距离矩阵。
    用 uint8 视图 + 查表做 popcount，避免逐位 Python 循环。
    """
    n, k = sigs.shape
    xor = sigs[:, None, :] ^ sigs[None, :, :]           # (N, N, K)
    as_bytes = xor.view(np.uint8).reshape(n, n, k, 8)   # 每个 uint64 拆 8 字节
    dist = _POPCOUNT[as_bytes].sum(axis=(2, 3))         # (N, N)
    return dist / float(k)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass
class DedupResult:
    keep_indices: list[int]
    drop_indices: list[int]
    n_clusters: int
    duplicate_rate: float


def dedup(
    sigs: np.ndarray,
    quality_scores: list[float] | None = None,
    threshold: float = 8.0,
) -> DedupResult:
    """
    threshold 是 64bit 上的平均汉明距离。
    经验值：<=8 判近似重复（约 12.5% 位不同），跨分辨率转码通常落在 2~6。
    阈值放太松会误杀同场景不同次的独立演示，必须实测调。
    """
    n = len(sigs)
    if n == 0:
        return DedupResult([], [], 0, 0.0)
    if quality_scores is None:
        quality_scores = [0.0] * n

    dist = hamming_matrix(sigs)
    uf = UnionFind(n)
    iu = np.triu_indices(n, k=1)
    for i, j in zip(*iu):
        if dist[i, j] <= threshold:
            uf.union(int(i), int(j))

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)

    keep, drop = [], []
    for members in clusters.values():
        best = max(members, key=lambda idx: quality_scores[idx])
        keep.append(best)
        drop.extend(m for m in members if m != best)

    return DedupResult(
        keep_indices=sorted(keep),
        drop_indices=sorted(drop),
        n_clusters=len(clusters),
        duplicate_rate=len(drop) / n if n else 0.0,
    )
