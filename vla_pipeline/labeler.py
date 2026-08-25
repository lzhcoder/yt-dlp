"""
弱标注层：把爬虫市场的平台原生记录转成管线可消费的候选 clip。

爬虫市场（Web Scraper API）返回的是平台原生字段，没有 scenario_type /
camera_pov / actions[]，也不做片段级预裁剪。这一层补三件事：

  1. 字段归一化：YouTube 的 video_length、TikTok 的 video_duration，统一成
     duration_ms；url / 网址 / video_url 统一成 source_url。接新平台只加
     候选键名，下游代码不动。
  2. 相关性判定：用关键词证据打分（环境词 + 主体词 + 动作词 - 负向词），
     分数低于阈值的整条丢弃。过滤发生在带宽之前。
  3. 固定窗口切分：按 max_duration_ms 均匀切，短于 min_duration_ms 的尾巴
     丢掉，单条源视频最多切 max_windows 个窗口。

所有标签都是**从文本推断的**，这一层没解码过任何一帧画面。所以每条记录都带
label_source="inferred_from_text" 和 clip_boundary="fixed_window"，
让下游自己决定信到什么程度。

纯标准库实现，不依赖任何第三方包。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from .taxonomy import ScenarioSpec, ENV_TYPES, ACTION_VOCAB


# ---------------------------------------------------------------------------
# 候选键表：各平台字段名不统一，按优先级尝试取值
# ---------------------------------------------------------------------------

_URL_KEYS = ("url", "video_url", "link", "网址", "source_url")
_DURATION_KEYS = ("video_length", "video_duration", "duration", "length", "时长")
_TITLE_KEYS = ("title", "video_title", "name")
_DESC_KEYS = ("description", "desc", "caption", "text")
_TAGS_KEYS = ("tags", "hashtags", "keywords", "topics")
_RES_KEYS = ("current_optimal_res", "resolution", "quality", "video_quality")
_FPS_KEYS = ("fps", "frame_rate", "framerate")


def _first(raw: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    """按候选键优先级取第一个非空值。"""
    for k in keys:
        v = raw.get(k)
        if v not in (None, "", []):
            return v
    return default


def _parse_duration_ms(v: Any) -> int:
    """把各种格式的时长统一成毫秒。输入可能是秒数(float)、'HH:MM:SS' 等。"""
    if v is None or v == "":
        return 0
    if isinstance(v, (int, float)):
        return int(float(v) * 1000)
    s = str(v).strip()
    # HH:MM:SS 或 MM:SS
    parts = s.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return 0
    if len(nums) == 3:
        return int((nums[0] * 3600 + nums[1] * 60 + nums[2]) * 1000)
    if len(nums) == 2:
        return int((nums[0] * 60 + nums[1]) * 1000)
    if len(nums) == 1:
        return int(nums[0] * 1000)
    return 0


def _parse_resolution(v: Any) -> tuple[int, int]:
    """从 '720p' / '1280x720' / '720' 解析出 (width, height)。未知返回 (0, 0)。"""
    if not v:
        return 0, 0
    s = str(v).strip().lower()
    # 1280x720
    m = re.match(r"(\d{3,5})\s*[x×]\s*(\d{3,5})", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    # 720p / 1080p
    m = re.match(r"(\d{3,4})p?", s)
    if m:
        h = int(m.group(1))
        w = int(h * 16 / 9) // 2 * 2
        return w, h
    return 0, 0


def _parse_fps(v: Any) -> float:
    if v is None or v == "":
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _join_text(*parts: Any) -> str:
    """把多个字段拼成一整段文本用于打分。列表会被展开。"""
    bits: list[str] = []
    for p in parts:
        if p is None:
            continue
        if isinstance(p, list):
            bits.extend(str(x) for x in p if x)
        else:
            s = str(p).strip()
            if s:
                bits.append(s)
    return " ".join(bits).lower()


# ---------------------------------------------------------------------------
# 归一化中间结构
# ---------------------------------------------------------------------------

@dataclass
class RawVideo:
    """平台原生记录归一化后的中间结构。"""
    source_url: str
    duration_ms: int
    text: str               # title + description + tags 拼接，已 lower
    platform: str           # "youtube" | "tiktok"
    fps: float              # 声称 fps，未知则 0
    width: int
    height: int
    raw: dict = field(default_factory=dict)   # 原始记录，保留用于 provenance


def adapt_youtube(raw: dict[str, Any]) -> RawVideo:
    """YouTube Videos 数据集的字段归一化。"""
    url = str(_first(raw, _URL_KEYS, "") or "")
    dur = _parse_duration_ms(_first(raw, _DURATION_KEYS))
    title = _first(raw, _TITLE_KEYS, "")
    desc = _first(raw, _DESC_KEYS, "")
    tags = _first(raw, _TAGS_KEYS, [])
    text = _join_text(title, desc, tags)
    w, h = _parse_resolution(_first(raw, _RES_KEYS))
    fps = _parse_fps(_first(raw, _FPS_KEYS))
    return RawVideo(
        source_url=url, duration_ms=dur, text=text, platform="youtube",
        fps=fps, width=w, height=h, raw=raw,
    )


def adapt_tiktok(raw: dict[str, Any]) -> RawVideo:
    """TikTok Posts 数据集的字段归一化。TikTok 的字段可能有中文键名。"""
    url = str(_first(raw, _URL_KEYS, "") or "")
    dur = _parse_duration_ms(_first(raw, _DURATION_KEYS))
    desc = _first(raw, _DESC_KEYS, "")
    tags = _first(raw, _TAGS_KEYS, [])
    text = _join_text(desc, tags)
    w, h = _parse_resolution(_first(raw, _RES_KEYS))
    fps = _parse_fps(_first(raw, _FPS_KEYS))
    return RawVideo(
        source_url=url, duration_ms=dur, text=text, platform="tiktok",
        fps=fps, width=w, height=h, raw=raw,
    )


ADAPTERS = {"youtube": adapt_youtube, "tiktok": adapt_tiktok}


def adapt(raw: dict[str, Any], platform: str) -> RawVideo:
    """按平台名分派到对应适配器。"""
    fn = ADAPTERS.get(platform)
    if fn is None:
        raise ValueError(f"不支持的平台 {platform!r}，已注册: {list(ADAPTERS)}")
    return fn(raw)


# ---------------------------------------------------------------------------
# 相关性判定：关键词证据打分
# ---------------------------------------------------------------------------

# 环境词表：ENV_TYPES -> 关键词列表（小写）
ENV_KEYWORDS: dict[str, tuple[str, ...]] = {
    "industrial_warehouse": ("warehouse", "storage", "shelf", "shelves", "rack",
                             "fulfillment", "logistics", "pallet", "拣货", "仓储", "货架"),
    "residential_kitchen": ("kitchen", "cooking", "counter", "stove", "sink",
                            "厨", "灶台", "水槽"),
    "office": ("office", "desk", "cubicle", "办公", "桌面"),
    "assembly_line": ("assembly line", "assembly", "conveyor", "factory",
                      "装配", "流水线", "传送"),
    "urban_street": ("urban", "street", "city", "road", "intersection",
                     "路口", "城市", "街道"),
    "highway": ("highway", "freeway", "expressway", "高速"),
    "parking_lot": ("parking", "parking lot", "garage", "停车场"),
    "construction_site": ("construction", "site", "building site", "工地", "施工"),
}

# 主体词表：视频里得有机器人才算数
SUBJECT_KEYWORDS = (
    "robot", "robotic", "arm", "gripper", "humanoid", "manipulator",
    "mechanical", "spider", "quadruped", "机器人", "机械臂", "夹爪",
)

# 动作词表：ACTION_VOCAB -> 近义词（小写），用于文本匹配
ACTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "reach": ("reach", "extend", "伸"),
    "grasp": ("grasp", "grip", "pick", "grab", "hold", "抓", "夹", "取"),
    "lift": ("lift", "raise", "elevate", "举", "抬"),
    "place": ("place", "put", "drop", "release", "放", "置"),
    "push": ("push", "shove", "推"),
    "pull": ("pull", "drag", "拉"),
    "wipe": ("wipe", "clean", "sweep", "擦", "抹", "清扫"),
    "pour": ("pour", "fill", "倒", "灌"),
    "insert": ("insert", "plug", "插"),
    "fasten": ("fasten", "screw", "bolt", "拧", "锁"),
    "align": ("align", "position", "对齐", "定位"),
    "sort": ("sort", "separate", "分拣", "分类"),
    "stack": ("stack", "pile", "码", "叠"),
    "walk": ("walk", "step", "stroll", "走", "步行"),
    "turn": ("turn", "rotate", "spin", "转"),
    "avoid": ("avoid", "dodge", "避"),
    "climb": ("climb", "ascend", "爬"),
    "open": ("open", "摊开", "开"),
    "close": ("close", "shut", "关"),
    "fold": ("fold", "叠", "折"),
    "press": ("press", "push button", "按"),
}

# 负向词表：游戏实况、仿真教程、AI 生成、开箱、访谈、宣传片
NEGATIVE_KEYWORDS = (
    "game", "gameplay", "walkthrough", "playthrough", "let's play",
    "tutorial", "how to use", "simulation", "simulator", "isaac sim",
    "mujoco", "pybullet", "sora", "ai generated", "ai-generated",
    "generated", "synthetic", "unboxing", "review", "interview",
    "promo", "trailer", "advert", "广告", "游戏", "教程", "仿真",
    "开箱", "测评", "采访",
)


def _count_hits(text: str, words: tuple[str, ...]) -> list[str]:
    """返回文本中命中的关键词列表（不去重，用于计分）。"""
    return [w for w in words if w in text]


def relevance_score(
    text: str,
    spec: ScenarioSpec,
) -> tuple[float, list[str], dict[str, int]]:
    """
    关键词证据打分。返回 (总分, 匹配到的动作列表, 分项计数)。

    score = env_hits × 2.0 + subject_hits × 1.5 + matched_actions × 1.5
            - negative_hits × 3.0
    """
    # 环境词：只算 spec 关心的环境
    env_hits: list[str] = []
    for env in spec.environments:
        env_hits.extend(_count_hits(text, ENV_KEYWORDS.get(env, ())))

    subject_hits = _count_hits(text, SUBJECT_KEYWORDS)

    # 动作词：检查 spec 期望的动作的近义词
    matched_actions: list[str] = []
    for action in spec.actions:
        kws = ACTION_KEYWORDS.get(action, ())
        if kws and any(kw in text for kw in kws):
            matched_actions.append(action)

    negative_hits = _count_hits(text, NEGATIVE_KEYWORDS)

    score = (
        min(len(env_hits), 3) * 2.0
        + min(len(subject_hits), 3) * 1.5
        + len(matched_actions) * 1.5
        - len(negative_hits) * 3.0
    )

    counts = {
        "env": len(env_hits),
        "subject": len(subject_hits),
        "actions": len(matched_actions),
        "negative": len(negative_hits),
    }
    return score, matched_actions, counts


# ---------------------------------------------------------------------------
# 场景/环境推断
# ---------------------------------------------------------------------------

def derive_env(text: str) -> str:
    """返回第一个命中的 ENV_TYPES，无命中返回 'unknown'。"""
    for env in ENV_TYPES:
        if any(kw in text for kw in ENV_KEYWORDS.get(env, ())):
            return env
    return "unknown"


def derive_scenario(text: str, spec: ScenarioSpec) -> str:
    """文本中命中 spec 的环境词就返回 spec.scenario_type，否则 'unknown'。"""
    for env in spec.environments:
        if any(kw in text for kw in ENV_KEYWORDS.get(env, ())):
            return spec.scenario_type
    return "unknown"


# ---------------------------------------------------------------------------
# 固定窗口切分
# ---------------------------------------------------------------------------

def segment_windows(
    video: RawVideo,
    spec: ScenarioSpec,
    max_windows: int = 3,
) -> list[tuple[int, int]]:
    """
    按 max_duration_ms 均匀切，短于 min_duration_ms 的尾巴丢掉。
    TikTok 整条视频作为一个窗口返回（短视频平台，整条通常 < 60s）。
    单条源视频最多切 max_windows 个窗口，防止一个来源主导数据集分布。
    """
    if video.duration_ms <= 0:
        return []

    # TikTok：整条视频 = 一个 clip
    if video.platform == "tiktok":
        if video.duration_ms >= spec.min_duration_ms:
            return [(0, video.duration_ms)]
        return []

    # YouTube：按 max_duration_ms 均匀切
    windows: list[tuple[int, int]] = []
    seg = spec.max_duration_ms
    start = 0
    while start < video.duration_ms and len(windows) < max_windows:
        end = min(start + seg, video.duration_ms)
        if end - start >= spec.min_duration_ms:
            windows.append((start, end))
        start = end
    return windows


# ---------------------------------------------------------------------------
# 主入口：一条平台原生记录 -> 0~N 条候选 clip
# ---------------------------------------------------------------------------

def label_record(
    raw: dict[str, Any],
    spec: ScenarioSpec,
    platform: str,
    min_score: float = 4.0,
    max_windows: int = 3,
) -> list[dict[str, Any]]:
    """
    把一条平台原生记录转成 0~N 条候选 clip（dict 格式，与 vendor_response.json 同构）。

    每条产出带：
      - label_source="inferred_from_text"
      - clip_boundary="fixed_window"（TikTok 为 "whole_video"）
      - relevance_score（float）
      - actions=[]（从文本推断出的动作列表）
      - camera_pov="third_person"（默认，无法从文本可靠推断）
      - scenario_type / env_context（从文本推断）
      - start_ms / end_ms / fps / source_url

    score < min_score 的整条丢弃（返回空列表）。
    """
    video = adapt(raw, platform)

    # 无 URL 的记录无法下载，直接丢
    if not video.source_url:
        return []

    # 时长为 0 无法切窗口，直接丢
    if video.duration_ms <= 0:
        return []

    score, matched_actions, _counts = relevance_score(video.text, spec)
    if score < min_score:
        return []

    scenario = derive_scenario(video.text, spec)
    env = derive_env(video.text)
    # 如果没命中 spec 的环境，但命中了其他已知环境，保留环境信息但不改 scenario
    if scenario == "unknown" and env == "unknown":
        # 连任何环境词都没命中，大概率不相关
        return []

    windows = segment_windows(video, spec, max_windows=max_windows)
    if not windows:
        return []

    boundary = "whole_video" if video.platform == "tiktok" else "fixed_window"
    fps = video.fps if video.fps > 0 else 30.0  # 未知时假设 30fps，探测层会核对

    clips: list[dict[str, Any]] = []
    for start_ms, end_ms in windows:
        clip = {
            "scenario_type": scenario,
            "env_context": env if env != "unknown" else spec.environments[0],
            "camera_pov": "third_person",
            "actions": matched_actions,           # 可能为空，下游用 --relaxed-actions
            "start_ms": start_ms,
            "end_ms": end_ms,
            "fps": fps,
            "source_url": video.source_url,
            "video_uri": "",                      # 待下载阶段回填
            "_label_source": "inferred_from_text",
            "_clip_boundary": boundary,
            "_relevance_score": round(score, 2),
            "_platform": platform,
            "_vendor": raw.get("_vendor", "brightdata"),
            "_snapshot_id": raw.get("_snapshot_id"),
            "_fetched_at": raw.get("_fetched_at"),
            # 保留声称的画质信息，探测层会核对
            "_declared_width": video.width,
            "_declared_height": video.height,
        }
        clips.append(clip)
    return clips


def label_batch(
    raws: list[dict[str, Any]],
    spec: ScenarioSpec,
    platform: str,
    min_score: float = 4.0,
    max_windows: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    批量标注。返回 (候选 clip 列表, 拒绝原因计数)。

    拒绝原因类别：
      no_url / no_duration / low_score / no_env / no_window
    """
    clips: list[dict[str, Any]] = []
    reject: dict[str, int] = {}

    for raw in raws:
        video = adapt(raw, platform)
        if not video.source_url:
            reject["no_url"] = reject.get("no_url", 0) + 1
            continue
        if video.duration_ms <= 0:
            reject["no_duration"] = reject.get("no_duration", 0) + 1
            continue

        score, matched_actions, _ = relevance_score(video.text, spec)
        if score < min_score:
            reject["low_score"] = reject.get("low_score", 0) + 1
            continue

        scenario = derive_scenario(video.text, spec)
        env = derive_env(video.text)
        if scenario == "unknown" and env == "unknown":
            reject["no_env"] = reject.get("no_env", 0) + 1
            continue

        windows = segment_windows(video, spec, max_windows=max_windows)
        if not windows:
            reject["no_window"] = reject.get("no_window", 0) + 1
            continue

        boundary = "whole_video" if video.platform == "tiktok" else "fixed_window"
        fps = video.fps if video.fps > 0 else 30.0

        for start_ms, end_ms in windows:
            clips.append({
                "scenario_type": scenario,
                "env_context": env if env != "unknown" else spec.environments[0],
                "camera_pov": "third_person",
                "actions": matched_actions,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "fps": fps,
                "source_url": video.source_url,
                "video_uri": "",
                "_label_source": "inferred_from_text",
                "_clip_boundary": boundary,
                "_relevance_score": round(score, 2),
                "_platform": platform,
                "_vendor": raw.get("_vendor", "brightdata"),
                "_snapshot_id": raw.get("_snapshot_id"),
                "_fetched_at": raw.get("_fetched_at"),
                "_declared_width": video.width,
                "_declared_height": video.height,
            })

    return clips, reject


# ---------------------------------------------------------------------------
# 按平台名匹配 dataset_id
# ---------------------------------------------------------------------------

# dataset_id 随账号权限变化，官方 GET /datasets/list 返回的就是当前账号可用的
# dataset。不要把可能属于另一个账号/产品版本的旧 ID 写死成默认 fallback——
# 开发者复制代码后最容易得到的是 401/404/invalid dataset，而不是成功运行。
# 正确做法：list_datasets() → 精确校验名匹配 → 环境变量覆盖 → 仍不存在就报错。

# 精确匹配模板：避免在长清单里被同名前缀的爬虫带偏。
# YouTube 账号下常有 "youtube video urls" / "yt videos" / "youtube discovery"
# 等十几个同名条目，只有 "videos posts" 才是我们要的带视频源的那个。
_DATASET_MATCH_KEYWORDS: dict[str, list[str]] = {
    # 优先级从高到低，命中即返回
    "youtube": ["videos posts", "video posts", "videos & posts", "videos posts + comments"],
    "tiktok": ["tiktok - posts", "tiktok posts"],
}


class DatasetNotAvailable(RuntimeError):
    """账号里没有匹配的爬虫，应去控制台开通，而不是静默回退到失效 ID。"""


def match_dataset_id(datasets: list[dict[str, Any]], platform: str) -> str:
    """从 list_datasets() 结果里按名字匹配平台对应的 dataset_id。

    匹配策略（从严到宽）：
      1. 精确词组匹配：YouTube 要 "videos posts"，避免被 "youtube video urls" /
         "yt videos" 这类同名爬虫带偏。
      2. 平台名兜底：只含 "youtube" / "tiktok" 但带 "posts" 的也认。
      3. 环境变量覆盖（BRIGHTDATA_YT_DATASET_ID / BRIGHTDATA_TT_DATASET_ID）。
      4. 都匹配不到 → 抛 DatasetNotAvailable，让开发者去开通权限，
         而不是静默用一个失效 ID 撞 API。
    """
    platform_lower = platform.lower()
    keywords = _DATASET_MATCH_KEYWORDS.get(platform_lower, [])

    # 1) 精确词组匹配
    for ds in datasets:
        name = str(ds.get("name", "")).lower()
        ds_id = str(ds.get("id", ds.get("dataset_id", "")))
        if not ds_id:
            continue
        if any(kw in name for kw in keywords):
            return ds_id

    # 2) 平台名 + posts 兜底
    for ds in datasets:
        name = str(ds.get("name", "")).lower()
        ds_id = str(ds.get("id", ds.get("dataset_id", "")))
        if not ds_id:
            continue
        if platform_lower in name and "post" in name:
            return ds_id

    # 3) 环境变量覆盖
    env_key = f"BRIGHTDATA_{platform_lower[:2].upper()}_DATASET_ID"
    env_val = os.environ.get(env_key)
    if env_val:
        return env_val

    # 4) 都没有 → 明确报错，不要用写死的旧 ID 静默撞 API
    raise DatasetNotAvailable(
        f"账号里未找到 {platform} 对应爬虫，请先在控制台开通，"
        f"或用环境变量 {env_key} 指定一个当前账号可用的 dataset_id。"
    )
