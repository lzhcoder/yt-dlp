"""
第二步：数据源接入

实现 Bright Data 的异步 snapshot 模式：
    POST /datasets/v3/trigger        -> {"snapshot_id": "s_xxx"}
    GET  /datasets/v3/progress/{id}  -> {"status": "running" | "ready" | "failed"}
    GET  /datasets/v3/snapshot/{id}  -> 结果数据
    POST /datasets/v3/deliver/{id}   -> 直接投递到 S3 / GCS / Azure / Snowflake

两个必须自己写对的地方：
  1. 退避。官方文档写明：同一 IP 在 5 分钟内累计 25 个 429 会被自动拉黑。
     所以 429 不能无脑重试，必须指数退避 + 尊重 Retry-After。
  2. 轮询上限。snapshot 卡在 running 是常态，必须有超时，否则脚本会挂一晚上。

MockBrightDataClient 让整条管线在没有 API Key 的情况下也能端到端跑通，
方便先把下游逻辑调好再花钱。
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Iterator

try:
    import requests
except ImportError:  # 只有真实调用才需要
    requests = None  # type: ignore

API_BASE = "https://api.brightdata.com"


class SnapshotTimeout(RuntimeError):
    pass


class SnapshotFailed(RuntimeError):
    pass


@dataclass
class TriggerResult:
    snapshot_id: str
    triggered_at: float


class BrightDataClient:
    """真实客户端。需要 BRIGHTDATA_API_KEY 环境变量。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = API_BASE,
        timeout: int = 60,
        max_429_backoff: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("BRIGHTDATA_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("缺少 BRIGHTDATA_API_KEY")
        if requests is None:
            raise RuntimeError("需要安装 requests: pip install requests")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_429_backoff = max_429_backoff
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        key_tail = self.api_key[-4:] if len(self.api_key) >= 4 else "?"
        print(f"[BrightDataClient] 真实模式, key=...{key_tail}, base={self.base_url}",
              flush=True)

    # -- 带退避的请求 --------------------------------------------------------

    def _request(self, method: str, path: str, *, max_retries: int = 6, **kw) -> Any:
        url = f"{self.base_url}{path}"
        delay = 2.0
        last_exc: Exception | None = None

        for attempt in range(max_retries):
            try:
                resp = self._session.request(method, url, timeout=self.timeout, **kw)
            except Exception as e:  # 网络抖动
                print(f"    {method} {path} 网络异常(第{attempt+1}次): {e}", flush=True)
                last_exc = e
                time.sleep(min(delay, self.max_429_backoff))
                delay *= 2
                continue

            if resp.status_code == 429:
                # 关键：尊重 Retry-After，并加抖动，避免整个集群同步重试
                retry_after = float(resp.headers.get("Retry-After", delay))
                sleep_s = min(retry_after, self.max_429_backoff) + random.uniform(0, 1.5)
                print(f"    {method} {path} 429 限流, 等 {sleep_s:.1f}s 后重试",
                      flush=True)
                time.sleep(sleep_s)
                delay *= 2
                continue

            if 500 <= resp.status_code < 600:
                print(f"    {method} {path} {resp.status_code} 服务端错误, "
                      f"等 {delay:.1f}s 后重试(第{attempt+1}次)", flush=True)
                time.sleep(min(delay, self.max_429_backoff))
                delay *= 2
                continue

            # 4xx（除 429）不重试：把响应体带出来，方便线上定位
            if 400 <= resp.status_code < 500:
                body = ""
                try:
                    body = resp.text[:800]
                except Exception:
                    pass
                raise RuntimeError(
                    f"{method} {path} 返回 {resp.status_code}: {body}"
                )

            resp.raise_for_status()
            if not resp.content:
                return None
            try:
                return resp.json()
            except ValueError:
                return resp.text

        raise RuntimeError(f"{method} {path} 重试 {max_retries} 次后仍失败: {last_exc}")

    # -- 三步工作流 ----------------------------------------------------------

    def trigger(
        self,
        dataset_id: str,
        payload: list[dict[str, Any]],
        *,
        fmt: str = "json",
        extra_params: dict[str, Any] | None = None,
    ) -> TriggerResult:
        """Define + Search：提交场景过滤条件，拿回 snapshot_id。"""
        params = {"dataset_id": dataset_id, "format": fmt}
        if extra_params:
            params.update(extra_params)
        print(f"    POST /datasets/v3/trigger dataset_id={dataset_id} "
              f"params={extra_params} body={payload[:1]}...", flush=True)
        data = self._request("POST", "/datasets/v3/trigger", params=params, json=payload)
        sid = data["snapshot_id"] if isinstance(data, dict) else str(data)
        print(f"    -> snapshot_id={sid}", flush=True)
        return TriggerResult(snapshot_id=sid, triggered_at=time.time())

    def wait_ready(
        self,
        snapshot_id: str,
        *,
        poll_interval: float = 10.0,
        max_wait: float = 3600.0,
    ) -> float:
        """轮询直到 ready，返回等待秒数。必须有 max_wait，否则会无限挂起。"""
        t0 = time.time()
        while True:
            state = self._request("GET", f"/datasets/v3/progress/{snapshot_id}")
            status = (state or {}).get("status", "unknown")
            elapsed = time.time() - t0
            # 每次轮询都打一行，让用户知道还在等、没死掉
            extra = ""
            if isinstance(state, dict):
                # 带上 BrightData 返回的进度百分比（如果有）
                for k in ("completion", "progress", "percent"):
                    if k in state:
                        extra = f" ({k}={state[k]})"
                        break
            print(f"    轮询 {snapshot_id}: status={status}, 已等 {elapsed:.0f}s{extra}",
                  flush=True)
            if status == "ready":
                return elapsed
            if status in ("failed", "error", "canceled"):
                raise SnapshotFailed(f"snapshot {snapshot_id} 状态 {status}")
            if elapsed > max_wait:
                raise SnapshotTimeout(f"snapshot {snapshot_id} 超过 {max_wait}s 仍未就绪")
            time.sleep(poll_interval)

    def download(self, snapshot_id: str, *, fmt: str = "json") -> list[dict[str, Any]]:
        """Extract：拉回结构化 metadata。注意结果有保留期，别拖太久。"""
        print(f"    GET /datasets/v3/snapshot/{snapshot_id} 下载结果...", flush=True)
        data = self._request(
            "GET", f"/datasets/v3/snapshot/{snapshot_id}", params={"format": fmt}
        )
        if isinstance(data, list):
            print(f"    -> {len(data)} 条记录", flush=True)
            return data
        if isinstance(data, str):  # ndjson
            rows = [json.loads(line) for line in data.splitlines() if line.strip()]
            print(f"    -> {len(rows)} 条记录 (ndjson)", flush=True)
            return rows
        print(f"    -> {1 if data else 0} 条记录", flush=True)
        return [data] if data else []

    def deliver_to_cloud(self, snapshot_id: str, delivery: dict[str, Any]) -> Any:
        """把结果直投客户私有云，数据不落第三方存储。"""
        return self._request("POST", f"/datasets/v3/deliver/{snapshot_id}", json=delivery)

    # -- 爬虫市场：发现模式 --------------------------------------------------

    def list_datasets(self) -> list[dict[str, Any]]:
        """GET /datasets/list — 返回账号可用的爬虫清单。
        用于运行时拿 dataset_id，不硬编码。"""
        data = self._request("GET", "/datasets/list")
        if isinstance(data, list):
            return data
        return [data] if data else []

    def discover(
        self,
        dataset_id: str,
        queries: list[str],
        discover_by: str = "keyword",
        limit_per_input: int = 20,
        **wait_kw,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """discover_new 模式：给检索词而非 URL，让爬虫自己去找新内容。

        按 Bright Data 文档，type / discover_by / limit_per_input 是 query 参数，
        检索词本身才进 JSON body。各平台 body 字段名不统一（YouTube 用 keyword，
        TikTok 用 search_keyword），这里用 discover_by 的值当 body 字段名，
        交给 Bright Data 侧做映射。
        """
        extra_params: dict[str, Any] = {
            "type": "discover_new",
            "discover_by": discover_by,
        }
        if limit_per_input:
            extra_params["limit_per_input"] = limit_per_input
        # body：每条只放检索词本身，字段名 = discover_by（keyword / search_keyword）
        payload: list[dict[str, Any]] = [{discover_by: q} for q in queries]
        return self.collect(dataset_id, payload, extra_params=extra_params, **wait_kw)

    # -- 组合调用 ------------------------------------------------------------

    def collect(
        self,
        dataset_id: str,
        payload: list[dict[str, Any]],
        *,
        extra_params: dict[str, Any] | None = None,
        **wait_kw,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        t0 = time.time()
        trig = self.trigger(dataset_id, payload, extra_params=extra_params)
        waited = self.wait_ready(trig.snapshot_id, **wait_kw)
        t1 = time.time()
        records = self.download(trig.snapshot_id)
        stats = {
            "snapshot_id": trig.snapshot_id,
            "wait_s": round(waited, 2),
            "download_s": round(time.time() - t1, 2),
            "total_s": round(time.time() - t0, 2),
            "records": len(records),
        }
        for r in records:
            r["_vendor"] = "brightdata"
            r["_snapshot_id"] = trig.snapshot_id
            r["_fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return records, stats


class MockBrightDataClient:
    """
    离线桩。接口与真实客户端一致，返回本地 fixture 生成的 metadata。
    用途：在不消耗额度、不联网的前提下，把下游校验/去重/打包全部跑通。
    """

    def __init__(self, fixture_dir: str) -> None:
        self.fixture_dir = fixture_dir

    def collect(self, dataset_id: str, payload: list[dict[str, Any]], **_) -> tuple[list[dict], dict]:
        t0 = time.time()
        path = os.path.join(self.fixture_dir, "vendor_response.json")
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            r["_vendor"] = "mock"
            r["_snapshot_id"] = "s_mock_local"
            r["_fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return records, {
            "snapshot_id": "s_mock_local",
            "wait_s": 0.0,
            "download_s": round(time.time() - t0, 3),
            "total_s": round(time.time() - t0, 3),
            "records": len(records),
        }

    def list_datasets(self) -> list[dict[str, Any]]:
        """返回硬编码的爬虫清单，模拟 GET /datasets/list。"""
        return [
            {"id": "gd_yt_videos", "name": "YouTube Videos Posts + Comments", "type": "scraper"},
            {"id": "gd_tt_posts", "name": "TikTok Posts", "type": "scraper"},
            {"id": "gd_tt_profiles", "name": "TikTok Profiles", "type": "scraper"},
            {"id": "gd_tt_shop", "name": "TikTok Shop", "type": "scraper"},
            {"id": "gd_tt_comments", "name": "TikTok Comments", "type": "scraper"},
        ]

    def discover(
        self,
        dataset_id: str,
        queries: list[str],
        discover_by: str = "keyword",
        limit_per_input: int = 20,
        **_,
    ) -> tuple[list[dict], dict]:
        """离线桩：优先读 discovery_response.json，没有就复用 vendor_response.json。"""
        t0 = time.time()
        disc_path = os.path.join(self.fixture_dir, "discovery_response.json")
        vend_path = os.path.join(self.fixture_dir, "vendor_response.json")
        path = disc_path if os.path.exists(disc_path) else vend_path
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        # 模拟 limit_per_input 截断
        if limit_per_input and len(records) > limit_per_input * len(queries):
            records = records[: limit_per_input * len(queries)]
        for r in records:
            r["_vendor"] = "mock"
            r["_snapshot_id"] = "s_mock_discovery"
            r["_fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            r["_dataset_id"] = dataset_id
        return records, {
            "snapshot_id": "s_mock_discovery",
            "wait_s": 0.0,
            "download_s": round(time.time() - t0, 3),
            "total_s": round(time.time() - t0, 3),
            "records": len(records),
        }


def get_client(fixture_dir: str | None = None) -> Any:
    """有 API Key 走真实链路，没有就走本地桩。"""
    if os.environ.get("BRIGHTDATA_API_KEY"):
        return BrightDataClient()
    if fixture_dir is None:
        raise RuntimeError("无 API Key 时必须提供 fixture_dir")
    return MockBrightDataClient(fixture_dir)
