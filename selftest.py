"""
接真实 API 之前的自检脚本。

用 mock fixture + 假 HTTP 层跑全部真实代码路径，验证管线完整性。
线上调试的时间成本比本地高一个数量级，能在本地发现的问题不要留到线上。

跑法：
    python make_fixtures.py --out ./fixtures --n 40
    python selftest.py

每项打印 ✓ / ✗，全过打印 ALL PASSED，有失败打印失败项并 sys.exit(1)。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# 确保能 import vla_pipeline（脚本可能在任意目录运行）
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from vla_pipeline.bd_client import get_client, MockBrightDataClient
from vla_pipeline.labeler import (
    adapt_youtube, adapt_tiktok, relevance_score, segment_windows,
    derive_scenario, derive_env, label_record, label_batch, match_dataset_id,
)
from vla_pipeline.schema import validate_batch, validate_batch_relaxed
from vla_pipeline.taxonomy import REGISTRY, WAREHOUSE_PICK_PLACE
from vla_pipeline.manifest import VLAClipDataset


FIXTURE_DIR = ROOT / "fixtures"
SELFTEST_OUT = ROOT / "selftest_out"
HARVEST_DIR = ROOT / "harvest_selftest"

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    mark = "✓" if ok else "✗"
    line = f"  {mark} {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    if ok:
        PASS += 1
    else:
        FAIL += 1


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), **kw)


# ---------------------------------------------------------------------------
# 测试项
# ---------------------------------------------------------------------------

def test_fixtures_exist() -> None:
    """1. make_fixtures.py 生成的 fixture 存在"""
    vendor = FIXTURE_DIR / "vendor_response.json"
    clips_dir = FIXTURE_DIR / "clips"
    if not vendor.exists():
        # 生成 40 条 fixture
        r = run([sys.executable, "make_fixtures.py", "--out", str(FIXTURE_DIR), "--n", "40"])
        if r.returncode != 0:
            check("fixture 生成", False, r.stderr[:200])
            return
    ok = vendor.exists() and clips_dir.exists() and any(clips_dir.iterdir())
    n = 0
    if vendor.exists():
        n = len(json.loads(vendor.read_text(encoding="utf-8")))
    check("fixture 存在（vendor_response.json + clips/）", ok, f"{n} 条 metadata")


def test_pipeline_e2e() -> None:
    """2. run_pipeline.py 跑通完整管线"""
    r = run([sys.executable, "run_pipeline.py",
             "--fixtures", str(FIXTURE_DIR), "--out", str(SELFTEST_OUT)])
    ok = r.returncode == 0
    check("run_pipeline.py 端到端", ok,
          (r.stdout[-150:] if ok else f"rc={r.returncode} " + r.stderr[-200:]).replace("\n", " "))


def test_manifest_exists() -> None:
    """3. manifest.jsonl 存在且行数 > 0"""
    m = SELFTEST_OUT / "manifest.jsonl"
    ok = m.exists()
    lines = 0
    if ok:
        lines = sum(1 for _ in m.open(encoding="utf-8") if _.strip())
    check("manifest.jsonl 存在且非空", ok and lines > 1, f"{lines} 行（含 header）")


def test_report_fields() -> None:
    """4. report.json 漏斗字段齐全"""
    rp = SELFTEST_OUT / "report.json"
    if not rp.exists():
        check("report.json 漏斗字段", False, "文件不存在")
        return
    data = json.loads(rp.read_text(encoding="utf-8"))
    funnel = data.get("funnel", {})
    needed = ["fetched", "schema_valid", "file_present", "probe_valid", "unique", "training_ready"]
    missing = [k for k in needed if k not in funnel]
    check("report.json 漏斗字段齐全", not missing,
          "缺失: " + ",".join(missing) if missing else ", ".join(f"{k}={funnel[k]}" for k in needed))


def test_dataset_load() -> None:
    """5. VLAClipDataset 能加载 manifest 且 len > 0"""
    m = SELFTEST_OUT / "manifest.jsonl"
    if not m.exists():
        check("VLAClipDataset 加载", False, "manifest 不存在")
        return
    ds = VLAClipDataset(m)
    check("VLAClipDataset 加载且 len > 0", len(ds) > 0, f"len={len(ds)}")


def test_list_datasets() -> None:
    """6. mock client 的 list_datasets() 返回非空"""
    client = MockBrightDataClient(str(FIXTURE_DIR))
    ds = client.list_datasets()
    check("list_datasets() 返回非空", len(ds) > 0, f"{len(ds)} 个爬虫")


def test_adapters() -> None:
    """7. adapt_youtube / adapt_tiktok 对样例记录能归一化"""
    yt_raw = {
        "url": "https://youtube.com/watch?v=abc",
        "video_length": 120,
        "title": "Warehouse robot grasp demo",
        "description": "robot arm pick and place",
        "tags": ["warehouse", "robot"],
        "current_optimal_res": "720p",
    }
    tt_raw = {
        "网址": "https://tiktok.com/@user/video/123",
        "video_duration": 30,
        "description": "kitchen robot wipe counter",
        "hashtags": ["robot", "kitchen"],
    }
    yt = adapt_youtube(yt_raw)
    tt = adapt_tiktok(tt_raw)
    ok = (yt.source_url == "https://youtube.com/watch?v=abc"
          and yt.duration_ms == 120_000
          and "warehouse" in yt.text
          and yt.height == 720
          and tt.source_url == "https://tiktok.com/@user/video/123"
          and tt.duration_ms == 30_000
          and "kitchen" in tt.text)
    check("adapt_youtube / adapt_tiktok 归一化", ok,
          f"yt(url={yt.source_url[:30]},dur={yt.duration_ms},h={yt.height}) "
          f"tt(url={tt.source_url[:30]},dur={tt.duration_ms})")


def test_relevance_score() -> None:
    """8. relevance_score 对正例高分，对负例低分"""
    spec = WAREHOUSE_PICK_PLACE
    pos = "warehouse robot arm grasp pick place demonstration"
    neg = "isaac sim simulation gameplay tutorial review"
    pos_score, pos_actions, _ = relevance_score(pos, spec)
    neg_score, neg_actions, _ = relevance_score(neg, spec)
    ok = pos_score > neg_score and pos_score >= 4.0 and neg_score < 0
    check("relevance_score 正例高分负例低分", ok,
          f"pos={pos_score:.1f}(actions={pos_actions}) neg={neg_score:.1f}(actions={neg_actions})")


def test_segment_windows() -> None:
    """9. segment_windows 对长视频切多个窗口，对短视频返回整条"""
    from vla_pipeline.labeler import RawVideo
    spec = WAREHOUSE_PICK_PLACE
    # 长视频 3 分钟 = 180000ms，max_duration_ms=45000 -> 最多 3 个窗口
    long_video = RawVideo(
        source_url="https://example.com/long", duration_ms=180_000,
        text="warehouse robot", platform="youtube", fps=30, width=1280, height=720, raw={},
    )
    # 短视频 20 秒，TikTok 整条
    short_tt = RawVideo(
        source_url="https://example.com/short", duration_ms=20_000,
        text="warehouse robot", platform="tiktok", fps=30, width=720, height=1280, raw={},
    )
    long_wins = segment_windows(long_video, spec, max_windows=3)
    short_wins = segment_windows(short_tt, spec, max_windows=3)
    ok = (len(long_wins) == 3 and long_wins[0][1] - long_wins[0][0] == 45_000
          and len(short_wins) == 1 and short_wins[0] == (0, 20_000))
    check("segment_windows 长视频多窗口 / TikTok 整条", ok,
          f"long={len(long_wins)}窗口 short={len(short_wins)}窗口")


def test_relaxed_validation() -> None:
    """10. validate_batch_relaxed 对 actions=[] 不拒绝"""
    spec = WAREHOUSE_PICK_PLACE
    # discovery 产出的典型记录：actions 可能为空
    relaxed_rec = {
        "scenario_type": "warehouse_pick_and_place",
        "env_context": "industrial_warehouse",
        "camera_pov": "third_person",
        "actions": [],                # 空 actions
        "start_ms": 0, "end_ms": 30_000, "fps": 30,
        "source_url": "https://example.com/v/test",
        "_label_source": "inferred_from_text",
        "_clip_boundary": "fixed_window",
    }
    # 严格模式应该拒绝（incomplete_actions）
    strict_passed, strict_reasons = validate_batch([relaxed_rec], spec)
    # 宽松模式应该通过
    relaxed_passed, relaxed_reasons = validate_batch_relaxed([relaxed_rec], spec)
    ok = (len(strict_passed) == 0 and len(relaxed_passed) == 1
          and relaxed_passed[0].provenance.get("label_source") == "inferred_from_text")
    check("validate_batch_relaxed 对 actions=[] 不拒绝", ok,
          f"strict={len(strict_passed)}(reasons={strict_reasons}) "
          f"relaxed={len(relaxed_passed)}(reasons={relaxed_reasons})")


def test_discovery_e2e() -> None:
    """11. run_discovery.py --no-download 在 mock 下跑通"""
    # 清理旧产出
    if HARVEST_DIR.exists():
        shutil.rmtree(HARVEST_DIR)
    r = run([sys.executable, "run_discovery.py",
             "--platforms", "youtube,tiktok",
             "--no-download", "--out", str(HARVEST_DIR),
             "--fixtures", str(FIXTURE_DIR)])
    ok = r.returncode == 0
    vendor_ok = (HARVEST_DIR / "vendor_response.json").exists()
    report_ok = (HARVEST_DIR / "discovery_report.json").exists()
    check("run_discovery.py --no-download (mock)", ok and vendor_ok and report_ok,
          (r.stdout[-120:] if ok else f"rc={r.returncode} " + r.stderr[-200:]).replace("\n", " "))


def main() -> None:
    print("=" * 60)
    print("VLA Pipeline 自检")
    print("=" * 60)

    # 确保 fixture 存在
    test_fixtures_exist()
    # 端到端管线
    test_pipeline_e2e()
    # 产出验证
    test_manifest_exists()
    test_report_fields()
    test_dataset_load()
    # discovery 模块
    test_list_datasets()
    test_adapters()
    test_relevance_score()
    test_segment_windows()
    test_relaxed_validation()
    # discovery 端到端
    test_discovery_e2e()

    print("\n" + "=" * 60)
    print(f"结果: {PASS} 通过, {FAIL} 失败")
    print("=" * 60)
    if FAIL > 0:
        print("SELFTEST FAILED")
        sys.exit(1)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
