"""
视频下载步骤：把供应商 metadata 里的远端视频按 [start_ms, end_ms] 时间窗下到本地。

主路径用 yt-dlp --download-sections 只下需要的片段（省带宽、帧精确）；
旧版 yt-dlp 或不支持分段的站点兜底成"下整段 → ffmpeg 截窗 → 删整段"。

幂等：video_uri 已是本地存在文件就直接返回，本地桩（mock）路径零改动。
yt-dlp 不在 PATH 上时退到 `python -m yt_dlp`，所以 pip --user 装的也能用。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from .schema import ClipRecord

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


def _yt_dlp_argv() -> list[str]:
    """yt-dlp 命令前缀。优先 PATH 上的 yt-dlp，否则退到 python -m yt_dlp。"""
    path = shutil.which("yt-dlp")
    if path:
        return [path]
    return [sys.executable, "-m", "yt_dlp"]


def _is_url(s: str | None) -> bool:
    if not s:
        return False
    p = urlparse(s)
    return p.scheme in ("http", "https") and bool(p.netloc)


def _resolve_url(rec: ClipRecord, url_field: str) -> str | None:
    """从记录里取可下载 URL。优先指定字段，退而取 video_uri，都不是 URL 返回 None。"""
    primary: str | None = rec.video_uri if url_field == "video_uri" else getattr(rec, url_field, None)
    if _is_url(primary):
        return primary
    if url_field != "video_uri" and _is_url(rec.video_uri):
        return rec.video_uri
    return None


def _local_ready(p: Path) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size > 0


def _check_yt_dlp() -> None:
    r = subprocess.run(_yt_dlp_argv() + ["--version"], capture_output=True, timeout=30, check=False)
    if r.returncode != 0:
        raise RuntimeError("yt-dlp 不可用，请安装：pip install yt-dlp 或 brew install yt-dlp")


def _download_section(url: str, dst: Path, start_s: float, end_s: float) -> bool:
    """主路径：只下 [start_s, end_s] 这一段，帧精确。"""
    cmd = _yt_dlp_argv() + [
        "--download-sections", f"*{start_s:.3f}-{end_s:.3f}",
        "--force-keyframes-at-cuts",
        "-f", "bv*+ba/b",
        "-o", str(dst),
        "--no-playlist",
        "--no-warnings",
        url,
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
    return r.returncode == 0 and _local_ready(dst)


def _download_full_and_extract(url: str, dst: Path, start_s: float, dur_s: float) -> bool:
    """兜底：下整段 → ffmpeg 截窗 → 删整段。-ss 在 -i 前做快速关键帧定位，再重编码。"""
    tmp = dst.with_suffix(".full.mp4")
    try:
        r = subprocess.run(
            _yt_dlp_argv() + ["-f", "bv*+ba/b", "-o", str(tmp), "--no-playlist", "--no-warnings", url],
            capture_output=True, timeout=1800, check=False,
        )
        if r.returncode != 0 or not _local_ready(tmp):
            return False
        fcmd = [
            FFMPEG, "-v", "error", "-y",
            "-ss", f"{start_s:.3f}", "-i", str(tmp),
            "-t", f"{dur_s:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-an",
            str(dst),
        ]
        fr = subprocess.run(fcmd, capture_output=True, timeout=600, check=False)
        return fr.returncode == 0 and _local_ready(dst)
    finally:
        tmp.unlink(missing_ok=True)


def download_clip(
    rec: ClipRecord,
    download_dir: Path,
    *,
    url_field: str = "source_url",
) -> tuple[ClipRecord, str | None]:
    """
    把 rec 的远端视频下到本地，回填 video_uri 为本地路径。
    返回 (rec, error)；error 为 None 表示成功（含"已本地存在"的跳过）。
    """
    # 幂等：已是本地存在文件 → 跳过
    if _local_ready(Path(rec.video_uri)):
        return rec, None

    url = _resolve_url(rec, url_field)
    if not url:
        return rec, "no_download_url"

    download_dir.mkdir(parents=True, exist_ok=True)
    dst = download_dir / f"{rec.clip_id}.mp4"
    if _local_ready(dst):  # 重跑：已下过
        rec.video_uri = str(dst)
        return rec, None

    start_s = rec.start_ms / 1000.0
    end_s = rec.end_ms / 1000.0
    dur_s = max(end_s - start_s, 0.1)

    if _download_section(url, dst, start_s, end_s):
        rec.video_uri = str(dst)
        return rec, None
    if _download_full_and_extract(url, dst, start_s, dur_s):
        rec.video_uri = str(dst)
        return rec, None
    return rec, "download_failed"


def download_batch(
    records: list[ClipRecord],
    download_dir: Path,
    *,
    url_field: str = "source_url",
    workers: int = 4,
) -> tuple[list[ClipRecord], dict[str, int]]:
    """
    并行下载。返回 (成功的 records, 失败原因计数)。
    成功 record 的 video_uri 已改写为本地路径；失败的原样保留。
    已是本地文件的记录直接归入成功、不触发 yt-dlp 检查（mock 路径零依赖）。
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    ok: list[ClipRecord] = []
    need_dl: list[ClipRecord] = []
    for rec in records:
        (ok if _local_ready(Path(rec.video_uri)) else need_dl).append(rec)

    failures: dict[str, int] = {}
    if need_dl:
        _check_yt_dlp()
        with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
            results = list(pool.map(
                lambda r: download_clip(r, download_dir, url_field=url_field), need_dl
            ))
        for rec, err in results:
            if err:
                failures[err] = failures.get(err, 0) + 1
            else:
                ok.append(rec)
    return ok, failures
