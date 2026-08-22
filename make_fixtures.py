"""
生成本地测试数据集，用于在没有 API Key 的情况下端到端验证整条管线。

刻意注入真实数据里一定会遇到的缺陷：
  - 近似重复（同素材不同分辨率/亮度转码）与完全重复
  - metadata 声称时长与实际文件不符
  - 帧率过低、分辨率过低
  - 动作标签残缺、场景标错、字段缺失
  - 文件损坏、文件缺失（下载失败）

跑法：
    python make_fixtures.py --out ./fixtures --n 120
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

GEO = ["DE", "US", "CN", "JP", "GB", "FR"]
POVS = ["third_person", "wrist_mounted"]
FULL_ACTIONS = ["reach", "grasp", "lift", "place"]


def gen_clip(path: Path, seed: int, duration_s: float, fps: int, height: int) -> bool:
    """用 life 滤镜生成结构差异明显的合成视频，保证不同 clip 的 pHash 可区分。"""
    width = int(height * 4 / 3) // 2 * 2
    vf = (
        f"life=s=160x120:mold=10:r={fps}:ratio=0.3:seed={seed}:"
        f"death_color=#101010:life_color=#e0e0e0,"
        f"scale={width}:{height}:flags=neighbor"
    )
    cmd = [
        FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", vf,
        "-t", f"{duration_s:.2f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
        str(path),
    ]
    return subprocess.run(cmd, capture_output=True, check=False).returncode == 0


def make_near_duplicate(src: Path, dst: Path) -> bool:
    """模拟"同一素材换个分辨率和亮度重新上传"——URL 和文件哈希都不同，但内容一样。"""
    cmd = [
        FFMPEG, "-v", "error", "-y", "-i", str(src),
        "-vf", "scale=-2:540,eq=brightness=0.06:contrast=1.08",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p",
        str(dst),
    ]
    return subprocess.run(cmd, capture_output=True, check=False).returncode == 0


def build(out_dir: Path, n: int, seed: int = 2026) -> None:
    rng = random.Random(seed)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    made: list[tuple[str, Path]] = []

    # ---- 1. 主体：正常片段 -------------------------------------------------
    n_clean = int(n * 0.62)
    for i in range(n_clean):
        dur = round(rng.uniform(4.0, 20.0), 2)
        fps = rng.choice([25, 30, 30, 30])
        height = rng.choice([480, 540, 720])
        name = f"clip_{i:04d}.mp4"
        p = clips_dir / name
        if not gen_clip(p, seed=1000 + i * 7, duration_s=dur, fps=fps, height=height):
            continue
        start = rng.randint(0, 200_000)
        records.append({
            "scenario_type": "warehouse_pick_and_place",
            "env_context": "industrial_warehouse",
            "camera_pov": rng.choice(POVS),
            "actions": list(FULL_ACTIONS),
            "start_ms": start,
            "end_ms": start + int(dur * 1000),
            "fps": fps,
            "geo_region": rng.choice(GEO),
            "source_url": f"https://example.com/watch?v=src{i:04d}",
            "video_uri": str(p),
        })
        made.append((name, p))

    idx = n_clean

    # ---- 2. 近似重复：取已有片段转码后重新登记 -----------------------------
    n_near = int(n * 0.11)
    for j in range(n_near):
        if not made:
            break
        src_name, src_path = made[rng.randrange(len(made))]
        name = f"clip_{idx:04d}.mp4"
        p = clips_dir / name
        if not make_near_duplicate(src_path, p):
            continue
        base = next(r for r in records if r["video_uri"] == str(src_path))
        dur_ms = base["end_ms"] - base["start_ms"]
        start = rng.randint(0, 200_000)
        records.append({
            **base,
            "start_ms": start,
            "end_ms": start + dur_ms,
            "source_url": f"https://mirror-site.example/v/{idx:04d}",
            "video_uri": str(p),
        })
        idx += 1

    # ---- 3. 完全重复：同一文件被登记两次 -----------------------------------
    for j in range(max(2, int(n * 0.03))):
        if not made:
            break
        src_name, src_path = made[rng.randrange(len(made))]
        name = f"clip_{idx:04d}.mp4"
        p = clips_dir / name
        shutil.copy(src_path, p)
        base = next(r for r in records if r["video_uri"] == str(src_path))
        records.append({
            **base,
            "source_url": f"https://repost.example/v/{idx:04d}",
            "video_uri": str(p),
        })
        idx += 1

    # ---- 4. metadata 声称时长与实际不符 ------------------------------------
    for j in range(max(2, int(n * 0.05))):
        dur = round(rng.uniform(2.0, 4.0), 2)
        name = f"clip_{idx:04d}.mp4"
        p = clips_dir / name
        gen_clip(p, seed=50_000 + idx, duration_s=dur, fps=30, height=480)
        start = rng.randint(0, 200_000)
        records.append({
            "scenario_type": "warehouse_pick_and_place",
            "env_context": "industrial_warehouse",
            "camera_pov": "third_person",
            "actions": list(FULL_ACTIONS),
            "start_ms": start,
            "end_ms": start + 15_000,      # 声称 15s，实际 2~4s
            "fps": 30,
            "geo_region": "US",
            "source_url": f"https://example.com/watch?v=drift{idx:04d}",
            "video_uri": str(p),
        })
        idx += 1

    # ---- 5. 帧率过低 / 分辨率过低 ------------------------------------------
    for j in range(max(2, int(n * 0.04))):
        low_fps = rng.choice([10, 12, 15])
        name = f"clip_{idx:04d}.mp4"
        p = clips_dir / name
        dur = round(rng.uniform(5.0, 12.0), 2)
        gen_clip(p, seed=60_000 + idx, duration_s=dur, fps=low_fps, height=360)
        start = rng.randint(0, 200_000)
        records.append({
            "scenario_type": "warehouse_pick_and_place",
            "env_context": "industrial_warehouse",
            "camera_pov": "third_person",
            "actions": list(FULL_ACTIONS),
            "start_ms": start,
            "end_ms": start + int(dur * 1000),
            "fps": low_fps,
            "geo_region": "CN",
            "source_url": f"https://example.com/watch?v=lowq{idx:04d}",
            "video_uri": str(p),
        })
        idx += 1

    # ---- 6. 动作标签残缺 ---------------------------------------------------
    for j in range(max(2, int(n * 0.06))):
        dur = round(rng.uniform(5.0, 15.0), 2)
        name = f"clip_{idx:04d}.mp4"
        p = clips_dir / name
        gen_clip(p, seed=70_000 + idx, duration_s=dur, fps=30, height=480)
        start = rng.randint(0, 200_000)
        records.append({
            "scenario_type": "warehouse_pick_and_place",
            "env_context": "industrial_warehouse",
            "camera_pov": "third_person",
            "actions": rng.choice([["grasp"], ["reach", "grasp"], ["lift", "place"]]),
            "start_ms": start,
            "end_ms": start + int(dur * 1000),
            "fps": 30,
            "geo_region": "JP",
            "source_url": f"https://example.com/watch?v=partial{idx:04d}",
            "video_uri": str(p),
        })
        idx += 1

    # ---- 7. 字段缺失 / 场景标错 --------------------------------------------
    n_noisy = max(4, int(n * 0.05))
    for j in range(n_noisy):
        dur = round(rng.uniform(5.0, 15.0), 2)
        name = f"clip_{idx:04d}.mp4"
        p = clips_dir / name
        gen_clip(p, seed=80_000 + idx, duration_s=dur, fps=30, height=480)
        start = rng.randint(0, 200_000)
        rec = {
            "scenario_type": "warehouse_pick_and_place",
            "env_context": "industrial_warehouse",
            "camera_pov": "third_person",
            "actions": list(FULL_ACTIONS),
            "start_ms": start,
            "end_ms": start + int(dur * 1000),
            "fps": 30,
            "source_url": f"https://example.com/watch?v=noisy{idx:04d}",
            "video_uri": str(p),
        }
        # 前一半：场景标错（供应商把厨房场景混进了仓储批次）
        # 后一半：字段缺失（camera_pov 为空）
        if j < n_noisy // 2:
            rec["scenario_type"] = "kitchen_manipulation_wipe"
        else:
            rec.pop("camera_pov")
        records.append(rec)
        idx += 1

    # ---- 8. 文件损坏 / 文件缺失 --------------------------------------------
    for j in range(max(1, int(n * 0.03))):
        name = f"clip_{idx:04d}.mp4"
        p = clips_dir / name
        p.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)   # 坏文件
        start = rng.randint(0, 200_000)
        records.append({
            "scenario_type": "warehouse_pick_and_place",
            "env_context": "industrial_warehouse",
            "camera_pov": "third_person",
            "actions": list(FULL_ACTIONS),
            "start_ms": start, "end_ms": start + 9000, "fps": 30,
            "geo_region": "GB",
            "source_url": f"https://example.com/watch?v=broken{idx:04d}",
            "video_uri": str(p),
        })
        idx += 1

    for j in range(max(1, int(n * 0.02))):
        name = f"clip_{idx:04d}.mp4"      # 只登记不落盘：模拟下载失败
        start = rng.randint(0, 200_000)
        records.append({
            "scenario_type": "warehouse_pick_and_place",
            "env_context": "industrial_warehouse",
            "camera_pov": "wrist_mounted",
            "actions": list(FULL_ACTIONS),
            "start_ms": start, "end_ms": start + 11_000, "fps": 30,
            "geo_region": "FR",
            "source_url": f"https://example.com/watch?v=missing{idx:04d}",
            "video_uri": str(clips_dir / name),
        })
        idx += 1

    rng.shuffle(records)
    (out_dir / "vendor_response.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已生成 {len(records)} 条 metadata，视频文件在 {clips_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./fixtures")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    build(Path(args.out), args.n, args.seed)
