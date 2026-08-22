"""
本地探测与预处理：metadata 说的和文件里实际是什么，必须分开验证。

最常见的坑：metadata 里 fps=30、时长 14.6s，但下载下来的文件是 24fps、11.2s。
如果不做这一步，训练时的时序对齐就是错的，而且很难查。

只依赖 ffprobe / ffmpeg 两个二进制，不需要 opencv。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .schema import ClipRecord
from .taxonomy import ScenarioSpec

FFPROBE = shutil.which("ffprobe") or "ffprobe"
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


@dataclass
class ProbeInfo:
    ok: bool
    duration_ms: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    codec: str = ""
    nb_frames: int = 0
    error: str = ""


def _parse_fps(rate: str) -> float:
    """ffprobe 的 r_frame_rate 是 '30000/1001' 这种分数形式。"""
    try:
        if "/" in rate:
            num, den = rate.split("/")
            return float(num) / float(den) if float(den) else 0.0
        return float(rate)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path: str | Path) -> ProbeInfo:
    """读取真实容器信息。文件损坏 / 零字节 / 无视频流都会在这里被挡下。"""
    path = str(path)
    cmd = [
        FFPROBE, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,codec_name,nb_read_packets:format=duration",
        "-count_packets", "-of", "json", path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
    except subprocess.TimeoutExpired:
        return ProbeInfo(ok=False, error="ffprobe_timeout")

    if out.returncode != 0:
        return ProbeInfo(ok=False, error="ffprobe_failed")

    try:
        meta = json.loads(out.stdout or b"{}")
    except ValueError:
        return ProbeInfo(ok=False, error="ffprobe_bad_json")

    streams = meta.get("streams") or []
    if not streams:
        return ProbeInfo(ok=False, error="no_video_stream")

    s = streams[0]
    fmt = meta.get("format") or {}
    try:
        duration_ms = int(float(fmt.get("duration", 0)) * 1000)
    except (TypeError, ValueError):
        duration_ms = 0

    return ProbeInfo(
        ok=True,
        duration_ms=duration_ms,
        fps=_parse_fps(s.get("r_frame_rate", "0/1")),
        width=int(s.get("width") or 0),
        height=int(s.get("height") or 0),
        codec=s.get("codec_name", ""),
        nb_frames=int(s.get("nb_read_packets") or 0),
    )


def check_against_metadata(
    rec: ClipRecord,
    info: ProbeInfo,
    spec: ScenarioSpec,
    *,
    duration_tolerance: float = 0.15,
) -> list[str]:
    """
    交叉核对 metadata 与真实文件。返回问题列表，空列表代表通过。
    duration_tolerance=0.15 表示允许 15% 的时长偏差（编码取整、关键帧对齐会有误差）。
    """
    problems: list[str] = []
    if not info.ok:
        return [f"probe_failed:{info.error}"]

    if info.height < spec.min_height:
        problems.append(f"resolution_too_low:{info.width}x{info.height}")
    if info.fps < spec.min_fps:
        problems.append(f"real_fps_too_low:{info.fps:.2f}")
    if info.duration_ms < spec.min_duration_ms:
        problems.append(f"real_duration_too_short:{info.duration_ms}ms")
    if info.nb_frames <= 1:
        problems.append("not_a_video")

    declared = rec.declared_duration_ms
    if declared > 0 and info.duration_ms > 0:
        drift = abs(info.duration_ms - declared) / declared
        if drift > duration_tolerance:
            problems.append(
                f"duration_mismatch:declared={declared}ms,real={info.duration_ms}ms"
            )

    if rec.fps > 0 and info.fps > 0 and abs(rec.fps - info.fps) / rec.fps > 0.05:
        problems.append(f"fps_mismatch:declared={rec.fps},real={info.fps:.2f}")

    return problems


def enrich(rec: ClipRecord, info: ProbeInfo) -> ClipRecord:
    """把实测值写回记录。之后所有下游一律用实测值，不用 metadata 声称的值。"""
    rec.duration_ms = info.duration_ms
    rec.width = info.width
    rec.height = info.height
    rec.codec = info.codec
    rec.fps = info.fps or rec.fps
    return rec


def normalize_fps(
    src: str | Path,
    dst: str | Path,
    *,
    target_fps: int = 30,
    target_height: int = 480,
    crf: int = 23,
) -> bool:
    """
    统一 fps 和短边分辨率。多来源混训时，帧率不一致会让 action chunk 的时间跨度不一致。
    宽度取 -2 保证是偶数，否则 libx264 会直接报错。
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-y", "-v", "error",
        "-i", str(src),
        "-vf", f"fps={target_fps},scale=-2:{target_height}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-an",  # VLA 视觉预训练一般用不到音轨，去掉省一半空间
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, check=False)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0
