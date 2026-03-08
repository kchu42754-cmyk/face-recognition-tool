# 人脸聚类工具

基于 InsightFace (ONNX) 的本地离线人脸聚类整理工具，支持 GPU 加速、缓存续跑，以及面向大规模数据的稳态聚类。

## 功能

- GPU 加速提取人脸特征
- 全离线处理，不上传图片
- 支持单图多人脸
- 缓存断点续跑
- 大规模数据自动切换到内存稳定的聚类后端

## 环境要求

| 组件 | 版本 |
| --- | --- |
| Python | 3.12+ |
| CUDA | 12.x |
| GPU | NVIDIA RTX 3060 或更高 |

## 安装依赖

```bash
pip install insightface onnxruntime-gpu numpy scikit-learn tqdm pillow opencv-python
```

## 路径配置

编辑 [`src/face_cluster.py`](../src/face_cluster.py) 中的目录配置：

```python
BASE_DIR = Path("/mnt/私密文件")
IMAGES_DIR = BASE_DIR / "images"
ORGANIZED_DIR = BASE_DIR / "organized"
FACES_DIR = BASE_DIR / "faces"
METADATA_DIR = BASE_DIR / "metadata"
```

## 运行方式

```bash
mkdir -p logs
nohup python3 src/face_cluster.py > logs/face_cluster.log 2>&1 &
tail -f logs/face_cluster.log
nvidia-smi
```

## 聚类策略

- 当人脸数不超过 `30,000` 时，使用 `AgglomerativeClustering`
- 当人脸数超过 `30,000` 时，自动切换到 `Birch`
- `Birch` 的引入是为了避免全量距离矩阵导致的内存爆炸

当前关键参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `FACE_TOLERANCE` | `0.6` | 小规模精确聚类的欧氏距离阈值 |
| `MAX_AGGLOMERATIVE_SIZE` | `30000` | 超过后切换到 `Birch` |
| `BIRCH_THRESHOLD` | `0.75` | 大规模聚类阈值 |
| `BIRCH_BRANCHING_FACTOR` | `50` | `Birch` 分支因子 |

## 支持的图片扩展名

- `.jpg`
- `.jpeg`
- `.jpe`
- `.png`
- `.gif`
- `.webp`
- `.bmp`
- `.tif`
- `.tiff`

扩展名匹配按大小写不敏感处理，例如 `.Jpg`、`.JPeG` 也会被扫描到。

## 输出结构

```text
/mnt/私密文件/
├── images/
├── faces/
│   └── face_cache_v3.pkl
├── organized/
│   ├── cluster_00000/
│   ├── cluster_00001/
│   └── no_face/
└── metadata/
    └── clusters.json
```

## 已知说明

- 少量损坏图片或 GIF 会在提取阶段被跳过，不会阻塞整批任务
- 大规模运行时请优先复用已有缓存，不要重复跑特征提取
- 如果目标文件系统不允许创建硬链接，脚本会自动回退到 `copy2`
- 2026-03-08 的 OOM 事故复盘见 [`INCIDENT-2026-03-08-oom.md`](./INCIDENT-2026-03-08-oom.md)
