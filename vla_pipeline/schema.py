"""
第三步：Schema 设计与校验

这一层做三件事：
  1. 校验供应商返回的 metadata（缺字段 / 类型错 / 值不在词表内）
  2. 归一化成内部统一 schema（换数据源不影响下游）
  3. 把校验失败的原因结构化记录下来——失败原因本身就是数据质量报告

关键设计：校验失败不抛异常，返回 (ok, reasons)。
一批 500 条里挂 30 条是常态，不能让整个 pipeline 崩掉。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

from .taxonomy import ScenarioSpec, ACTION_VOCAB, CAMERA_POVS, ENV_TYPES

# 供应商原生 schema 的必填字段
# 对齐 Bright Data VLA 页面公开的 per-clip metadata 结构
REQUIRED_FIELDS = (
    "scenario_type", "env_context", "camera_pov",
    "actions", "start_ms", "end_ms", "fps",
)
OPTIONAL_FIELDS = ("geo_region", "source_url", "title", "language")

SCHEMA_VERSION = "yt-dlp-clip/1.2"


@dataclass
class ClipRecord:
    """内部统一 schema。下游（dedup / manifest / Dataset）只认这个结构。"""

    clip_id: str
    video_uri: str
    scenario_type: str
    env_context: str
    camera_pov: str
    actions: list[str]
    start_ms: int
    end_ms: int
    fps: float
    geo_region: str | None = None
    source_url: str | None = None

    # 由本地探测阶段回填
    duration_ms: int | None = None
    width: int | None = None
    height: int | None = None
    codec: str | None = None
    phash: str | None = None

    schema_version: str = SCHEMA_VERSION
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def declared_duration_ms(self) -> int:
        """metadata 声称的时长。跟 ffprobe 实测时长对不上，就是数据质量问题。"""
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    record: ClipRecord | None = None

    def fail(self, reason: str) -> "ValidationResult":
        self.ok = False
        self.reasons.append(reason)
        return self


def make_clip_id(raw: dict[str, Any]) -> str:
    """
    稳定 ID：同一个源视频的同一个时间窗，无论抓多少次都得到同一个 clip_id。
    这是断点续传和跨批次去重的基础，不能用 uuid4。
    """
    key = "|".join([
        str(raw.get("source_url") or raw.get("video_uri") or ""),
        str(raw.get("start_ms", "")),
        str(raw.get("end_ms", "")),
    ])
    return "clip_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def validate_and_normalize(
    raw: dict[str, Any],
    spec: ScenarioSpec,
    video_uri: str | None = None,
) -> ValidationResult:
    """把供应商返回的一条 metadata 校验并转成 ClipRecord。"""
    res = ValidationResult(ok=True)

    # 1) 必填字段
    for f in REQUIRED_FIELDS:
        if raw.get(f) in (None, "", []):
            res.fail(f"missing_field:{f}")
    if not res.ok:
        return res

    # 2) 类型与取值范围
    try:
        start_ms = int(raw["start_ms"])
        end_ms = int(raw["end_ms"])
        fps = float(raw["fps"])
    except (TypeError, ValueError):
        return res.fail("type_error:timestamps_or_fps")

    if end_ms <= start_ms:
        res.fail("invalid_window:end<=start")
    if fps < spec.min_fps:
        res.fail(f"fps_too_low:{fps}")

    dur = end_ms - start_ms
    if dur < spec.min_duration_ms:
        res.fail(f"too_short:{dur}ms")
    if dur > spec.max_duration_ms:
        res.fail(f"too_long:{dur}ms")

    # 3) 受控词表校验——这一步能挡住绝大多数"看起来有值其实没用"的脏数据
    if raw["scenario_type"] != spec.scenario_type:
        res.fail(f"scenario_mismatch:{raw['scenario_type']}")
    if raw["camera_pov"] not in CAMERA_POVS:
        res.fail(f"unknown_pov:{raw['camera_pov']}")
    elif raw["camera_pov"] not in spec.camera_povs:
        res.fail(f"pov_out_of_scope:{raw['camera_pov']}")
    if raw["env_context"] not in ENV_TYPES:
        res.fail(f"unknown_env:{raw['env_context']}")

    actions = raw.get("actions") or []
    if not isinstance(actions, list):
        return res.fail("type_error:actions_not_list")
    unknown = [a for a in actions if a not in ACTION_VOCAB]
    if unknown:
        res.fail(f"unknown_actions:{','.join(unknown)}")

    # 4) 动作完整性：taxonomy 要求 reach/grasp/lift/place，只标了 grasp 的片段
    #    对 pick-and-place 训练来说是残缺的
    missing_actions = [a for a in spec.actions if a not in actions]
    if missing_actions:
        res.fail(f"incomplete_actions:missing={','.join(missing_actions)}")

    if not res.ok:
        return res

    res.record = ClipRecord(
        clip_id=make_clip_id(raw),
        video_uri=video_uri or raw.get("video_uri") or "",
        scenario_type=raw["scenario_type"],
        env_context=raw["env_context"],
        camera_pov=raw["camera_pov"],
        actions=list(actions),
        start_ms=start_ms,
        end_ms=end_ms,
        fps=fps,
        geo_region=raw.get("geo_region"),
        source_url=raw.get("source_url"),
        provenance={
            "vendor": raw.get("_vendor", "unknown"),
            "snapshot_id": raw.get("_snapshot_id"),
            "fetched_at": raw.get("_fetched_at"),
            "label_source": raw.get("_label_source", "vendor_provided"),
            "clip_boundary": raw.get("_clip_boundary", "pre_cut"),
        },
    )
    return res


def validate_batch(
    raws: Iterable[dict[str, Any]],
    spec: ScenarioSpec,
) -> tuple[list[ClipRecord], dict[str, int]]:
    """批量校验，返回 (通过的记录, 失败原因计数)。失败原因计数就是质量报告。"""
    passed: list[ClipRecord] = []
    reason_counter: dict[str, int] = {}

    for raw in raws:
        r = validate_and_normalize(raw, spec, video_uri=raw.get("video_uri"))
        if r.ok and r.record is not None:
            passed.append(r.record)
        else:
            for reason in r.reasons:
                # 只保留原因类别，不保留具体值，否则计数表会炸开
                key = reason.split(":")[0]
                reason_counter[key] = reason_counter.get(key, 0) + 1

    return passed, reason_counter


def _validate_relaxed(
    raw: dict[str, Any],
    spec: ScenarioSpec,
    video_uri: str | None = None,
) -> ValidationResult:
    """宽松校验：actions 允许为空，不做 incomplete_actions 检查。

    用于 discovery 产出的弱标注候选数据——它们是文本推断的，actions 可能为空，
    强行要求动作完整会把整批数据挡掉。其余检查（必填字段、词表、时长、fps）
    照常，保证下游 ffprobe / dedup 仍能拿到可信的 metadata。
    """
    res = ValidationResult(ok=True)

    # 1) 必填字段——actions 单独处理，允许空列表
    for f in REQUIRED_FIELDS:
        if f == "actions":
            v = raw.get(f)
            if v is None or v == "":
                res.fail(f"missing_field:{f}")
            continue
        if raw.get(f) in (None, "", []):
            res.fail(f"missing_field:{f}")
    if not res.ok:
        return res

    # 2) 类型与取值范围
    try:
        start_ms = int(raw["start_ms"])
        end_ms = int(raw["end_ms"])
        fps = float(raw["fps"])
    except (TypeError, ValueError):
        return res.fail("type_error:timestamps_or_fps")

    if end_ms <= start_ms:
        res.fail("invalid_window:end<=start")
    if fps < spec.min_fps:
        res.fail(f"fps_too_low:{fps}")

    dur = end_ms - start_ms
    if dur < spec.min_duration_ms:
        res.fail(f"too_short:{dur}ms")
    if dur > spec.max_duration_ms:
        res.fail(f"too_long:{dur}ms")

    # 3) 受控词表校验
    if raw["scenario_type"] != spec.scenario_type:
        res.fail(f"scenario_mismatch:{raw['scenario_type']}")
    if raw["camera_pov"] not in CAMERA_POVS:
        res.fail(f"unknown_pov:{raw['camera_pov']}")
    elif raw["camera_pov"] not in spec.camera_povs:
        res.fail(f"pov_out_of_scope:{raw['camera_pov']}")
    if raw["env_context"] not in ENV_TYPES:
        res.fail(f"unknown_env:{raw['env_context']}")

    actions = raw.get("actions") or []
    if not isinstance(actions, list):
        return res.fail("type_error:actions_not_list")
    unknown = [a for a in actions if a not in ACTION_VOCAB]
    if unknown:
        res.fail(f"unknown_actions:{','.join(unknown)}")

    # 4) 动作完整性检查：宽松模式下跳过——actions 可能为空或部分

    if not res.ok:
        return res

    res.record = ClipRecord(
        clip_id=make_clip_id(raw),
        video_uri=video_uri or raw.get("video_uri") or "",
        scenario_type=raw["scenario_type"],
        env_context=raw["env_context"],
        camera_pov=raw["camera_pov"],
        actions=list(actions),
        start_ms=start_ms,
        end_ms=end_ms,
        fps=fps,
        geo_region=raw.get("geo_region"),
        source_url=raw.get("source_url"),
        provenance={
            "vendor": raw.get("_vendor", "unknown"),
            "snapshot_id": raw.get("_snapshot_id"),
            "fetched_at": raw.get("_fetched_at"),
            "label_source": raw.get("_label_source", "inferred_from_text"),
            "clip_boundary": raw.get("_clip_boundary", "fixed_window"),
        },
    )
    return res


def validate_batch_relaxed(
    raws: Iterable[dict[str, Any]],
    spec: ScenarioSpec,
) -> tuple[list[ClipRecord], dict[str, int]]:
    """宽松批量校验：actions 允许为空，不做 incomplete_actions 检查。

    用于 discovery 产出的弱标注数据（label_source="inferred_from_text"）。
    """
    passed: list[ClipRecord] = []
    reason_counter: dict[str, int] = {}

    for raw in raws:
        r = _validate_relaxed(raw, spec, video_uri=raw.get("video_uri"))
        if r.ok and r.record is not None:
            passed.append(r.record)
        else:
            for reason in r.reasons:
                key = reason.split(":")[0]
                reason_counter[key] = reason_counter.get(key, 0) + 1

    return passed, reason_counter


if __name__ == "__main__":
    from .taxonomy import WAREHOUSE_PICK_PLACE

    good = {
        "scenario_type": "warehouse_pick_and_place",
        "env_context": "industrial_warehouse",
        "camera_pov": "wrist_mounted",
        "actions": ["reach", "grasp", "lift", "place"],
        "start_ms": 5200, "end_ms": 19800, "fps": 30, "geo_region": "DE",
        "source_url": "https://example.com/v/abc",
    }
    bad = dict(good, actions=["grasp"], fps=8, end_ms=6000)

    ok_recs, reasons = validate_batch([good, bad], WAREHOUSE_PICK_PLACE)
    print("通过:", len(ok_recs), "| clip_id:", ok_recs[0].clip_id)
    print("失败原因:", json.dumps(reasons, ensure_ascii=False))
