# vla-pipeline

配套《2026 年 VLA 训练数据管线从零搭建》一文的最小可运行实现。

一条从「场景定义」到「PyTorch Dataset」的完整视频数据管线，
带 Schema 校验、ffprobe 交叉核对、pHash 近似去重和产出率漏斗报告。

## 依赖

- Python 3.10+
- `ffmpeg` / `ffprobe`（必需，两个二进制）
- `numpy`（必需）
- `requests`（只有接真实 API 时需要）
- `torch`（可选；没装时 `VLAClipDataset` 退化为普通可索引对象，manifest 校验照常跑）

```bash
sudo apt-get install -y ffmpeg          # 或 brew install ffmpeg
pip install numpy requests
```

## 离线跑通（不需要 API Key，不联网）

```bash
python make_fixtures.py --out ./fixtures --n 120
python run_pipeline.py --fixtures ./fixtures --out ./dataset --cost 48.0
```

`make_fixtures.py` 会合成 100+ 条测试片段，并按真实比例注入缺陷：
近似重复、完全重复、时长与 metadata 不符、帧率过低、动作标签残缺、
场景标错、字段缺失、文件损坏、文件缺失。

输出：

- `dataset/manifest.jsonl` — 训练侧唯一入口，第一行是数据集 header
- `dataset/report.json` — 漏斗与拒绝原因统计

## 接真实数据源

```bash
export BRIGHTDATA_API_KEY=xxxxx
python run_pipeline.py --dataset-id gd_xxxxxxxx --out ./dataset
```

`vla_pipeline/bd_client.py` 实现的是 Bright Data 的异步 snapshot 模式
（`/datasets/v3/trigger` → `/datasets/v3/progress/{id}` → `/datasets/v3/snapshot/{id}`），
含 429 指数退避与轮询超时。换供应商时只需替换这一个文件，
`taxonomy.to_filter_params()` 是唯一需要改字段映射的地方。

## 范围说明

- 本仓库验证的是**管线逻辑**：取数 → 校验 → 探测 → 去重 → 打包 → 消费。
  离线模式下的产出率是针对合成缺陷数据集的，**不是任何供应商的真实评测结果**。
- 供应商侧的场景检索精度、切片边界准确率需要用你自己的账号和场景实测，
  `report.json` 的漏斗字段就是为了记录这些数字设计的。
- `actions[]` 是语义动作标注，不是机器人动作向量。这条管线产出的是
  预训练/表征学习可用的视频数据，不能直接当作 VLA policy 的 action label。

## 模块

| 文件 | 职责 |
| --- | --- |
| `vla_pipeline/taxonomy.py` | 场景分类体系；受控词表既是过滤条件也是校验白名单 |
| `vla_pipeline/schema.py` | metadata 校验与归一化，失败原因结构化；含宽松校验模式 |
| `vla_pipeline/bd_client.py` | 异步 snapshot 客户端 + 爬虫市场 discovery 模式 + 离线 mock |
| `vla_pipeline/labeler.py` | 弱标注层：平台字段归一化 + 关键词相关性打分 + 固定窗口切分 |
| `vla_pipeline/preprocess.py` | ffprobe 探测、metadata 交叉核对、fps/分辨率归一化 |
| `vla_pipeline/dedup.py` | pHash 签名 + 并查集近似去重（仅需 ffmpeg + numpy） |
| `vla_pipeline/manifest.py` | manifest 读写、产出率报告、Dataset |
| `vla_pipeline/downloader.py` | yt-dlp 视频下载，按时间窗截取，幂等 |
| `make_fixtures.py` | 合成带缺陷的测试数据集 |
| `run_pipeline.py` | 端到端编排与报告（支持 `--relaxed-actions` 消费 discovery 产出） |
| `run_discovery.py` | 爬虫市场降级路径：发现 → 打分 → 切分 → 下载 |
| `selftest.py` | 接真实 API 前的自检脚本 |
