"""
端到端管线：取数 → Schema 校验 → 本地探测 → 去重 → manifest → Dataset

跑法（无需 API Key，用本地 fixture）：
    python make_fixtures.py --out ./fixtures --n 120
    python run_pipeline.py --fixtures ./fixtures --out ./dataset

接真实数据源：
    export BRIGHTDATA_API_KEY=...
    python run_pipeline.py --dataset-id gd_xxxxxxxx --out ./dataset

输出：
    dataset/manifest.jsonl   训练侧唯一入口
    dataset/report.json      漏斗与拒绝原因统计
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from vla_pipeline.bd_client import get_client, BrightDataClient
from vla_pipeline.dedup import clip_signature, dedup
from vla_pipeline.manifest import VLAClipDataset, YieldReport, write_manifest
from vla_pipeline.preprocess import check_against_metadata, enrich, probe
from vla_pipeline.schema import validate_batch, validate_batch_relaxed
from vla_pipeline.taxonomy import REGISTRY


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="warehouse_pick_and_place")
    ap.add_argument("--fixtures", default="./fixtures")
    ap.add_argument("--dataset-id", default=None, help="真实调用时的 Bright Data dataset_id")
    ap.add_argument("--out", default="./dataset")
    ap.add_argument("--dedup-threshold", type=float, default=8.0)
    ap.add_argument("--cost", type=float, default=0.0, help="本批总花费，用于算单条有效成本")
    ap.add_argument("--workers", type=int, default=8, help="抽帧并发数")
    ap.add_argument("--download-dir", default=None, help="视频下载目录，默认 {out}/downloads")
    ap.add_argument("--download-workers", type=int, default=4, help="下载并发数")
    ap.add_argument("--url-field", default="source_url", help="记录里取下载 URL 的字段名")
    ap.add_argument("--no-download", action="store_true", help="跳过下载步骤（视频已在本地）")
    ap.add_argument("--relaxed-actions", action="store_true",
                    help="允许 actions 为空（用于 discovery 产出的弱标注数据）")
    args = ap.parse_args()

    spec = REGISTRY.get(args.scenario)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = YieldReport()

    # ---- ① 取数 ----------------------------------------------------------
    # 三种取数路径，按优先级自动选择：
    #   a) 真实链路 + fixtures 目录里有 vendor_response.json → 直接读本地，
    #      不再调 API（供 run_discovery.py --no-download 产出的候选消费）。
    #   b) 真实链路 + 没有 --dataset-id → 没法调 API，报错让用户补参数或先跑 discovery。
    #   c) 真实链路 + 有 --dataset-id → 走 BrightData trigger/collect。
    # 无 API Key 时 get_client() 返回 mock，mock.collect() 自己读本地 vendor_response.json。
    t0 = time.time()
    client = get_client(fixture_dir=args.fixtures)
    is_real = isinstance(client, BrightDataClient)

    local_vendor = Path(args.fixtures) / "vendor_response.json"
    if is_real and local_vendor.exists():
        # 路径 a：discovery 已拉好数据，这里只做下游加工，不重复花钱
        with open(local_vendor, "r", encoding="utf-8") as f:
            raw_records = json.load(f)
        fetch_stats = {
            "snapshot_id": "local",
            "wait_s": 0.0,
            "download_s": round(time.time() - t0, 3),
            "total_s": round(time.time() - t0, 3),
            "records": len(raw_records),
            "source": f"local:{local_vendor}",
        }
        print(f"① 读取本地 {local_vendor}（{len(raw_records)} 条，跳过 API 取数）")
    elif is_real and not args.dataset_id:
        # 路径 b：真实模式但既没有本地文件也没给 dataset_id
        raise SystemExit(
            "真实模式下取数需要以下之一：\n"
            "  1. 先跑 run_discovery.py --no-download 产出 vendor_response.json，"
            "再用 --fixtures 指向那个目录；\n"
            "  2. 或显式传 --dataset-id <真实爬虫ID> 直接调 BrightData。"
        )
    else:
        # 路径 c / mock：走 client.collect()
        payload = [spec.to_filter_params()]
        raw_records, fetch_stats = client.collect(args.dataset_id or "mock", payload)
    report.fetched = len(raw_records)
    report.timings["fetch_s"] = round(time.time() - t0, 2)
    print(f"① 取回 {report.fetched} 条 "
          f"(snapshot={fetch_stats['snapshot_id']}, {report.timings['fetch_s']}s)")

    # ---- ② Schema 校验 ---------------------------------------------------
    t0 = time.time()
    if args.relaxed_actions:
        records, reasons = validate_batch_relaxed(raw_records, spec)
        print("  (宽松校验：actions 允许为空，跳过 incomplete_actions 检查)")
    else:
        records, reasons = validate_batch(raw_records, spec)
    report.schema_valid = len(records)
    report.reject_reasons.update(reasons)
    report.timings["schema_s"] = round(time.time() - t0, 2)
    print(f"② Schema 通过 {report.schema_valid} 条")

    # ---- ②.5 下载视频到本地（仅真实链路；mock 的 video_uri 已是本地文件）----
    if is_real and not args.no_download and records:
        t0 = time.time()
        from vla_pipeline.downloader import download_batch
        dl_dir = Path(args.download_dir) if args.download_dir else out_dir / "downloads"
        records, dl_failures = download_batch(
            records, dl_dir, url_field=args.url_field, workers=args.download_workers
        )
        for k, v in dl_failures.items():
            report.reject_reasons[k] = report.reject_reasons.get(k, 0) + v
        report.timings["download_s"] = round(time.time() - t0, 2)
        dropped = sum(dl_failures.values())
        print(f"②.5 下载完成 {len(records)} 条"
              + (f"（失败 {dropped}: {dl_failures}）" if dl_failures else "")
              + f" {report.timings['download_s']}s")

    # ---- ③④ 文件探测 -----------------------------------------------------
    t0 = time.time()
    probed = []
    for rec in records:
        p = Path(rec.video_uri)
        if not p.exists() or p.stat().st_size == 0:
            report.reject_reasons["file_missing"] = \
                report.reject_reasons.get("file_missing", 0) + 1
            continue
        report.file_present += 1

        info = probe(p)
        problems = check_against_metadata(rec, info, spec)
        if problems:
            for pr in problems:
                key = pr.split(":")[0]
                report.reject_reasons[key] = report.reject_reasons.get(key, 0) + 1
            continue
        probed.append(enrich(rec, info))

    report.probe_valid = len(probed)
    report.timings["probe_s"] = round(time.time() - t0, 2)
    print(f"③ 文件存在 {report.file_present} 条 | ④ 探测通过 {report.probe_valid} 条 "
          f"({report.timings['probe_s']}s)")

    # ---- ⑤ 去重 ----------------------------------------------------------
    t0 = time.time()
    if probed:
        # 抽帧算 pHash 是整条管线的瓶颈（每条一次 ffmpeg 调用）。
        # ffmpeg 是子进程，不受 GIL 限制，用线程池并行即可线性加速。
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            sigs = np.array(list(pool.map(
                lambda r: clip_signature(r.video_uri, r.duration_ms or 0), probed
            )))
        # 质量分：分辨率优先，同簇保留最清晰的一条
        quality = [float((r.width or 0) * (r.height or 0)) for r in probed]
        dd = dedup(sigs, quality_scores=quality, threshold=args.dedup_threshold)
        unique_records = [probed[i] for i in dd.keep_indices]
        for i in dd.keep_indices:
            probed[i].phash = format(int(sigs[i][0]), "016x")
        report.reject_reasons["near_duplicate"] = len(dd.drop_indices)
        dup_rate = dd.duplicate_rate
    else:
        unique_records, dup_rate = [], 0.0

    report.unique = len(unique_records)
    report.training_ready = len(unique_records)
    report.timings["dedup_s"] = round(time.time() - t0, 2)
    print(f"⑤ 去重后 {report.unique} 条（重复率 {dup_rate:.1%}，"
          f"{report.timings['dedup_s']}s）")

    # ---- ⑥ 打包 ----------------------------------------------------------
    manifest_path = out_dir / "manifest.jsonl"
    header = write_manifest(
        unique_records, manifest_path,
        dataset_name=f"{spec.scenario_type}_web",
        version=time.strftime("v%Y%m%d"),
    )

    print("\n" + report.render())
    if args.cost > 0:
        print(f"{'单条有效成本'.ljust(18)}  "
              f"{report.cost_per_usable_clip(args.cost):.4f} / clip")

    print("\n拒绝原因分布：")
    for k, v in sorted(report.reject_reasons.items(), key=lambda x: -x[1]):
        if v:
            print(f"  {k:<28} {v}")

    (out_dir / "report.json").write_text(
        json.dumps({
            "funnel": {
                "fetched": report.fetched,
                "schema_valid": report.schema_valid,
                "file_present": report.file_present,
                "probe_valid": report.probe_valid,
                "unique": report.unique,
                "training_ready": report.training_ready,
                "usable_yield": round(report.usable_yield, 4),
            },
            "reject_reasons": report.reject_reasons,
            "timings": report.timings,
            "manifest_header": header,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---- ⑦ 验证训练侧能消费 ----------------------------------------------
    ds = VLAClipDataset(manifest_path)
    print(f"\nDataset 长度 = {len(ds)}  (torch 可用: {ds.has_torch})")
    if len(ds):
        print("dataset[0] =", json.dumps(ds[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
