"""
打包与消费：manifest + Dataset

manifest 是训练侧唯一入口。任何时候都不要让训练脚本去 glob 目录——
目录会变，manifest 有版本号，可复现。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

from .schema import ClipRecord, SCHEMA_VERSION


@dataclass
class YieldReport:
    """漏斗报告。这张表比"我们有多少 TB 数据"有用得多。"""

    fetched: int = 0
    schema_valid: int = 0
    file_present: int = 0
    probe_valid: int = 0
    unique: int = 0
    training_ready: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    # discovery 模式（爬虫市场降级路径）的漏斗前置阶段
    # 常规路径（供应商直出场景标签）这两个值为 0，render 时自动跳过
    discovered: int = 0       # discovery 模式取回的原生记录数
    labeled: int = 0          # 通过相关性打分 + 切分的候选数

    @property
    def usable_yield(self) -> float:
        return self.training_ready / self.fetched if self.fetched else 0.0

    def cost_per_usable_clip(self, total_cost: float) -> float:
        return total_cost / self.training_ready if self.training_ready else float("inf")

    def render(self) -> str:
        rows: list[tuple[str, int]] = []
        # discovery 模式才显示发现 + 标注两行
        if self.discovered > 0:
            rows.append(("⓪ discovery 返回", self.discovered))
            rows.append(("① 弱标注候选", self.labeled))
            # discovery 模式下 fetched = labeled（候选进入校验）
            base = self.labeled or self.fetched or 1
            rows.append(("② Schema 通过", self.schema_valid))
            rows.append(("③ 文件存在", self.file_present))
            rows.append(("④ 探测通过", self.probe_valid))
            rows.append(("⑤ 去重后", self.unique))
            rows.append(("⑥ 训练可用", self.training_ready))
        else:
            base = self.fetched or 1
            rows = [
                ("① 供应商返回", self.fetched),
                ("② Schema 通过", self.schema_valid),
                ("③ 文件存在", self.file_present),
                ("④ 探测通过", self.probe_valid),
                ("⑤ 去重后", self.unique),
                ("⑥ 训练可用", self.training_ready),
            ]
        width = max(len(r[0]) for r in rows)
        lines = [f"{'阶段'.ljust(width)}  {'数量':>6}  {'留存率':>7}"]
        lines.append("-" * (width + 18))
        for name, val in rows:
            lines.append(f"{name.ljust(width)}  {val:>6}  {val / base:>6.1%}")
        lines.append("-" * (width + 18))
        lines.append(f"{'raw → usable yield'.ljust(width)}  {self.usable_yield:>13.1%}")
        return "\n".join(lines)


def write_manifest(
    records: Iterable[ClipRecord],
    path: str | Path,
    *,
    dataset_name: str,
    version: str,
) -> dict[str, Any]:
    """写 JSONL manifest，第一行是 header（数据集元信息），其余每行一条样本。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)

    header = {
        "_type": "manifest_header",
        "dataset_name": dataset_name,
        "version": version,
        "schema_version": SCHEMA_VERSION,
        "n_samples": len(records),
        "scenario_distribution": dict(Counter(r.scenario_type for r in records)),
        "pov_distribution": dict(Counter(r.camera_pov for r in records)),
        "geo_distribution": dict(Counter(r.geo_region or "unknown" for r in records)),
        "label_source_distribution": dict(Counter(
            r.provenance.get("label_source", "unknown") for r in records
        )),
        "total_duration_s": round(sum((r.duration_ms or 0) for r in records) / 1000, 1),
    }

    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return header


def read_manifest(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    header: dict[str, Any] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if i == 0 and obj.get("_type") == "manifest_header":
                header = obj
            else:
                rows.append(obj)
    return header, rows


# ---------------------------------------------------------------------------
# 训练侧入口
# ---------------------------------------------------------------------------

try:
    from torch.utils.data import Dataset as _TorchDataset  # type: ignore
    _HAS_TORCH = True
except ImportError:  # 没装 torch 也能跑，方便在数据机器上单独验证 manifest
    _TorchDataset = object  # type: ignore
    _HAS_TORCH = False


class VLAClipDataset(_TorchDataset):  # type: ignore[misc]
    """
    最小可用的 Dataset。这里刻意不做解码——
    真实训练里视频解码应该交给 decord / torchcodec / DALI，
    这个类的职责只是证明 manifest 能被正确消费，以及暴露每条样本的字段。
    """

    def __init__(self, manifest_path: str | Path, transform=None) -> None:
        self.header, self.rows = read_manifest(manifest_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.rows[idx]
        sample = {
            "video_uri": r["video_uri"],
            "clip_id": r["clip_id"],
            "scenario": r["scenario_type"],
            "actions": r["actions"],          # 语义动作标签，不是机器人动作向量
            "start_ms": r["start_ms"],
            "end_ms": r["end_ms"],
            "fps": r["fps"],
            "camera_pov": r["camera_pov"],
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    @property
    def has_torch(self) -> bool:
        return _HAS_TORCH
