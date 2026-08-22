"""
第一步：场景分类体系（Scenario Taxonomy）

核心原则：先定义你要什么数据，再去取数据。
Taxonomy 有两个用途：
  1. 生成数据源的过滤条件（Bright Data Filter API / Scraper 的 query 参数）
  2. 作为下游校验的白名单——不在 taxonomy 里的值一律判为脏数据

纯标准库实现，不依赖任何第三方包。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# 受控词表（Controlled Vocabulary）
# 这些常量同时是"过滤条件"和"校验白名单"，只有一份定义，避免两边不一致
# ---------------------------------------------------------------------------

TASK_FAMILIES = ("manipulation", "locomotion", "interaction", "driving")

ACTION_VOCAB = (
    # manipulation
    "reach", "grasp", "lift", "place", "push", "pull", "wipe", "pour",
    "insert", "fasten", "align", "sort", "stack",
    # locomotion
    "walk", "turn", "avoid", "climb",
    # interaction
    "open", "close", "fold", "press",
)

CAMERA_POVS = ("third_person", "wrist_mounted", "overhead", "egocentric", "vehicle_mounted")

ENV_TYPES = (
    "industrial_warehouse", "residential_kitchen", "office", "assembly_line",
    "urban_street", "highway", "parking_lot", "construction_site",
)

LIGHTING = ("normal", "low_light", "backlit", "night")


@dataclass(frozen=True)
class ScenarioSpec:
    """一个可执行的场景定义。它既是需求文档，也是过滤参数，也是校验规则。"""

    scenario_type: str                      # 全局唯一 slug，下游一切以它为准
    task_family: str
    environments: tuple[str, ...]
    actions: tuple[str, ...]                # 期望出现的动作序列（语义层）
    camera_povs: tuple[str, ...]
    lighting: tuple[str, ...] = ("normal",)

    # 质量门槛：低于这些值的片段直接丢弃，不进入训练集
    min_duration_ms: int = 3_000
    max_duration_ms: int = 60_000
    min_height: int = 480
    min_fps: float = 20.0

    # 采样目标：用于计算这一批还差多少条
    target_clips: int = 500

    def __post_init__(self) -> None:
        """构造即校验。宁可在第 0 秒报错，也不要在 10 万条数据之后才发现拼错了。"""
        if self.task_family not in TASK_FAMILIES:
            raise ValueError(f"未知 task_family: {self.task_family}")
        for a in self.actions:
            if a not in ACTION_VOCAB:
                raise ValueError(f"动作 {a!r} 不在受控词表内，请先扩充 ACTION_VOCAB")
        for p in self.camera_povs:
            if p not in CAMERA_POVS:
                raise ValueError(f"未知 camera_pov: {p}")
        for e in self.environments:
            if e not in ENV_TYPES:
                raise ValueError(f"未知 env_context: {e}")
        if self.min_duration_ms >= self.max_duration_ms:
            raise ValueError("min_duration_ms 必须小于 max_duration_ms")

    # -- 输出为数据源过滤条件 -------------------------------------------------

    def to_filter_params(self) -> dict[str, Any]:
        """
        转成供应商侧的过滤条件。
        字段名按 Bright Data VLA 页面公开的 metadata 字段对齐
        （scenario_type / env_context / camera_pov / actions / fps 等）。
        换供应商时只需要改这个函数，taxonomy 本身不动。
        """
        return {
            "scenario_type": self.scenario_type,
            "env_context": list(self.environments),
            "camera_pov": list(self.camera_povs),
            "action_type": list(self.actions),
            "lighting": list(self.lighting),
            "min_duration_ms": self.min_duration_ms,
            "max_duration_ms": self.max_duration_ms,
            "min_resolution_height": self.min_height,
            "limit": self.target_clips,
        }

    def to_search_queries(self) -> list[str]:
        """
        降级路径：如果拿不到场景级过滤 API，只有关键词检索入口，
        用 taxonomy 生成一组关键词，至少保证检索意图是被显式定义过的。
        """
        env_words = {
            "industrial_warehouse": "warehouse",
            "residential_kitchen": "kitchen",
            "assembly_line": "assembly line",
            "office": "office",
            "urban_street": "urban street",
            "highway": "highway",
            "parking_lot": "parking lot",
            "construction_site": "construction site",
        }
        queries = []
        for env in self.environments:
            base = env_words.get(env, env.replace("_", " "))
            queries.append(f"{base} robot {' '.join(self.actions[:2])}")
            queries.append(f"{base} {self.actions[0]} {self.actions[-1]} demonstration")
        return queries

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaxonomyRegistry:
    """全部场景的注册表。建议纳入版本控制，改动走 code review。"""

    version: str
    scenarios: dict[str, ScenarioSpec] = field(default_factory=dict)

    def register(self, spec: ScenarioSpec) -> ScenarioSpec:
        if spec.scenario_type in self.scenarios:
            raise ValueError(f"scenario_type 重复: {spec.scenario_type}")
        self.scenarios[spec.scenario_type] = spec
        return spec

    def get(self, scenario_type: str) -> ScenarioSpec:
        if scenario_type not in self.scenarios:
            raise KeyError(f"未注册的 scenario_type: {scenario_type}")
        return self.scenarios[scenario_type]


# ---------------------------------------------------------------------------
# 本文全程使用的示例：仓储抓取-放置
# ---------------------------------------------------------------------------

REGISTRY = TaxonomyRegistry(version="2026.08.1")

WAREHOUSE_PICK_PLACE = REGISTRY.register(
    ScenarioSpec(
        scenario_type="warehouse_pick_and_place",
        task_family="manipulation",
        environments=("industrial_warehouse",),
        actions=("reach", "grasp", "lift", "place"),
        camera_povs=("third_person", "wrist_mounted"),
        lighting=("normal", "low_light"),
        min_duration_ms=3_000,
        max_duration_ms=45_000,
        min_height=480,
        min_fps=20.0,
        target_clips=500,
    )
)

KITCHEN_WIPE = REGISTRY.register(
    ScenarioSpec(
        scenario_type="kitchen_manipulation_wipe",
        task_family="manipulation",
        environments=("residential_kitchen",),
        actions=("reach", "grasp", "wipe", "place"),
        camera_povs=("third_person", "overhead"),
        lighting=("normal", "low_light"),
        target_clips=300,
    )
)


if __name__ == "__main__":
    import json

    spec = REGISTRY.get("warehouse_pick_and_place")
    print("过滤条件：")
    print(json.dumps(spec.to_filter_params(), ensure_ascii=False, indent=2))
    print("\n降级关键词：")
    for q in spec.to_search_queries():
        print(" -", q)
