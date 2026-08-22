"""
爬虫市场降级路径的编排入口。

当账号只有按站点划分的通用爬虫（YouTube / TikTok），没有场景级 VLA Feed 时，
用这个脚本完成"场景发现 + 相关性判定 + 片段切分"，产出与 fixture 同构的
vendor_response.json，下游 run_pipeline.py 一行不改即可消费。

文章里的三条命令都必须能用：

    # 先看账号里实际有哪些爬虫
    python run_discovery.py --list-datasets

    # 第一次只做发现和筛选，不花下载带宽
    python run_discovery.py --platforms youtube,tiktok --no-download --out ./harvest

    # 调好 --min-score 再放量
    python run_discovery.py \
        --scenario warehouse_pick_and_place \
        --platforms youtube,tiktok \
        --per-query 20 --max-clips 60 \
        --out ./harvest

    # 产出与 fixture 同构，下游一行不改
    python run_pipeline.py --fixtures ./harvest --out ./dataset --relaxed-actions

无 BRIGHTDATA_API_KEY 时自动走 mock：读 fixtures/discovery_response.json
（没有就复用 vendor_response.json），下游逻辑全跑真实代码。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from vla_pipeline.bd_client import get_client, BrightDataClient
from vla_pipeline.labeler import (
    adapt, label_batch, match_dataset_id, relevance_score, segment_windows,
    derive_scenario, derive_env,
)
from vla_pipeline.taxonomy import REGISTRY


def _print_datasets(datasets: list[dict]) -> None:
    """打印爬虫清单表格。"""
    if not datasets:
        print("（账号下没有可用爬虫）")
        return
    print(f"{'dataset_id':<20} {'name':<40} {'type'}")
    print("-" * 72)
    for ds in datasets:
        did = str(ds.get("id", ds.get("dataset_id", "")))
        name = str(ds.get("name", ""))[:40]
        dtype = str(ds.get("type", ""))
        print(f"{did:<20} {name:<40} {dtype}")


def _run_platform(
    client,
    platform: str,
    spec,
    datasets: list[dict],
    per_query: int,
    min_score: float,
    max_clips: int,
    fixture_dir: str,
) -> tuple[list[dict], dict[str, int], int]:
    """跑单个平台的发现 → 标注 → 切分，返回 (候选 clips, 拒绝原因, 原始记录数)。"""
    dataset_id = match_dataset_id(datasets, platform)
    if not dataset_id:
        print(f"  [{platform}] 未在账号爬虫清单里匹配到对应 dataset_id，跳过")
        return [], {}, 0

    queries = spec.to_search_queries()
    print(f"  [{platform}] dataset_id={dataset_id}, 检索词 {len(queries)} 个: {queries}")

    # discover：给检索词，让爬虫自己去找新内容
    raws, stats = client.discover(
        dataset_id, queries,
        discover_by="keyword",
        limit_per_input=per_query,
    )
    print(f"  [{platform}] discovery 返回 {len(raws)} 条 (snapshot={stats.get('snapshot_id')})")

    # 弱标注：归一化 → 打分 → 切分
    clips, reject = label_batch(
        raws, spec, platform,
        min_score=min_score,
        max_windows=3,
    )
    print(f"  [{platform}] 弱标注候选 {len(clips)} 条，拒绝 {sum(reject.values())} 条: {reject}")
    return clips, reject, len(raws)


def _download_clips(clips: list[dict], out_dir: Path, workers: int) -> tuple[list[dict], dict[str, int]]:
    """用 yt-dlp 把候选 clip 的源视频按时间窗下到本地，回填 video_uri。"""
    from vla_pipeline.downloader import download_clip
    from vla_pipeline.schema import ClipRecord

    dl_dir = out_dir / "downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    ok: list[dict] = []
    failures: dict[str, int] = {}

    # 把 clip dict 临时包装成 ClipRecord 复用 download_clip
    for clip in clips:
        rec = ClipRecord(
            clip_id=f"disc_{abs(hash(clip['source_url'] + str(clip['start_ms']))):016x}"[:20],
            video_uri=clip.get("video_uri", ""),
            scenario_type=clip["scenario_type"],
            env_context=clip["env_context"],
            camera_pov=clip["camera_pov"],
            actions=clip["actions"],
            start_ms=clip["start_ms"],
            end_ms=clip["end_ms"],
            fps=clip["fps"],
            source_url=clip["source_url"],
        )
        rec2, err = download_clip(rec, dl_dir, url_field="source_url")
        if err:
            failures[err] = failures.get(err, 0) + 1
        else:
            clip["video_uri"] = rec2.video_uri
            ok.append(clip)
    return ok, failures


def main() -> None:
    ap = argparse.ArgumentParser(description="爬虫市场降级路径：发现 → 打分 → 切分 → 下载")
    ap.add_argument("--list-datasets", action="store_true",
                    help="列出账号可用的爬虫清单，不抓数据")
    ap.add_argument("--scenario", default="warehouse_pick_and_place",
                    help="场景 slug（从 REGISTRY 取 spec）")
    ap.add_argument("--platforms", default="youtube,tiktok",
                    help="逗号分隔的平台名，如 youtube,tiktok")
    ap.add_argument("--per-query", type=int, default=20,
                    help="每个检索词最多返回多少条（传给 discover 的 limit_per_input）")
    ap.add_argument("--max-clips", type=int, default=60,
                    help="候选总数上限")
    ap.add_argument("--min-score", type=float, default=4.0,
                    help="相关性最低分，低于此分整条丢弃")
    ap.add_argument("--no-download", action="store_true",
                    help="只做发现+打分+切分，不下载视频，产出 vendor_response.json 同构文件")
    ap.add_argument("--download-workers", type=int, default=4)
    ap.add_argument("--fixtures", default="./fixtures",
                    help="mock 模式下读 fixture 的目录")
    ap.add_argument("--out", default="./harvest", help="输出目录")
    args = ap.parse_args()

    spec = REGISTRY.get(args.scenario)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 有 key 走真实，无 key 走 mock
    client = get_client(fixture_dir=args.fixtures)
    is_real = isinstance(client, BrightDataClient)

    # ---- --list-datasets：只列爬虫清单 ----------------------------------------
    if args.list_datasets:
        print(f"账号可用爬虫清单 ({'真实' if is_real else 'mock'}):\n")
        _print_datasets(client.list_datasets())
        return

    # ---- 拿 dataset_id 清单 ---------------------------------------------------
    datasets = client.list_datasets()
    if not datasets:
        print("账号下没有可用爬虫，无法进入 discovery 模式")
        return

    platforms = [p.strip() for p in args.platforms.split(",") if p.strip()]
    print(f"场景: {spec.scenario_type} | 平台: {platforms} | min_score={args.min_score}")

    # ---- 对每个平台跑 discovery → 标注 → 切分 ---------------------------------
    t0 = time.time()
    all_clips: list[dict] = []
    all_reject: dict[str, int] = {}
    total_discovered = 0

    for platform in platforms:
        clips, reject, n_raw = _run_platform(
            client, platform, spec, datasets,
            per_query=args.per_query,
            min_score=args.min_score,
            max_clips=args.max_clips,
            fixture_dir=args.fixtures,
        )
        all_clips.extend(clips)
        for k, v in reject.items():
            all_reject[k] = all_reject.get(k, 0) + v
        total_discovered += n_raw

    # ---- URL 级去重 -----------------------------------------------------------
    seen_urls: set[str] = set()
    deduped: list[dict] = []
    dup_count = 0
    for clip in all_clips:
        url = clip["source_url"]
        if url in seen_urls:
            dup_count += 1
            continue
        seen_urls.add(url)
        deduped.append(clip)
    all_reject["url_duplicate"] = dup_count
    print(f"\nURL 级去重: {len(all_clips)} -> {len(deduped)} (重复 {dup_count})")

    # ---- 截断到 --max-clips ---------------------------------------------------
    if len(deduped) > args.max_clips:
        truncated = len(deduped) - args.max_clips
        deduped = deduped[: args.max_clips]
        print(f"截断到 --max-clips={args.max_clips}，丢弃 {truncated} 条")

    # ---- 下载（除非 --no-download）--------------------------------------------
    download_failures: dict[str, int] = {}
    if args.no_download:
        print("--no-download：跳过下载步骤，产出候选 metadata 供 run_pipeline.py 消费")
    elif is_real and deduped:
        print(f"\n下载 {len(deduped)} 条候选视频...")
        dl_t0 = time.time()
        deduped, download_failures = _download_clips(deduped, out_dir, args.download_workers)
        print(f"下载完成 {len(deduped)} 条"
              + (f"（失败 {sum(download_failures.values())}: {download_failures}）"
                 if download_failures else "")
              + f" {time.time() - dl_t0:.1f}s")

    # ---- 写 vendor_response.json（与 fixture 同构）---------------------------
    vendor_path = out_dir / "vendor_response.json"
    vendor_path.write_text(
        json.dumps(deduped, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n已写 {vendor_path}（{len(deduped)} 条候选，与 fixture 同构）")

    # ---- 写 discovery_report.json（发现漏斗 + 拒绝原因）-----------------------
    report = {
        "scenario": spec.scenario_type,
        "platforms": platforms,
        "min_score": args.min_score,
        "per_query": args.per_query,
        "max_clips": args.max_clips,
        "funnel": {
            "discovered": total_discovered,
            "labeled": len(all_clips),
            "url_deduped": len(deduped) + dup_count,
            "final": len(deduped),
        },
        "reject_reasons": all_reject,
        "download_failures": download_failures,
        "download_skipped": args.no_download,
        "is_real_api": is_real,
        "timings_s": round(time.time() - t0, 2),
        "label_source": "inferred_from_text",
        "clip_boundary": "fixed_window",
        "next_step": (
            "python run_pipeline.py --fixtures " + str(out_dir)
            + " --out ./dataset --relaxed-actions"
        ),
    }
    report_path = out_dir / "discovery_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已写 {report_path}")

    # ---- 打印漏斗摘要 ---------------------------------------------------------
    print("\n发现漏斗：")
    print(f"  discovery 返回     {total_discovered}")
    print(f"  弱标注候选         {len(all_clips)}")
    print(f"  URL 级去重后       {len(deduped) + dup_count}  (重复 {dup_count})")
    print(f"  最终产出           {len(deduped)}")
    if all_reject:
        print("\n拒绝原因分布：")
        for k, v in sorted(all_reject.items(), key=lambda x: -x[1]):
            if v:
                print(f"  {k:<24} {v}")
    print(f"\n下游消费：\n  {report['next_step']}")


if __name__ == "__main__":
    main()
