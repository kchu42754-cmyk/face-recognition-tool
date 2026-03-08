# 人脸识别聚类工具

基于 InsightFace (ONNX) GPU 加速的本地离线人脸聚类整理工具。

## 功能特性

- ✅ GPU 加速 (CUDA)
- ✅ 纯离线处理，不上云
- ✅ 支持多人脸检测
- ✅ 分批聚类，避免 OOM
- ✅ 增量处理（断点续传）

## 环境要求

| 组件 | 版本 |
|------|------|
| Python | 3.12+ |
| CUDA | 12.6 |
| GPU | NVIDIA RTX 3060+ |

## 安装依赖

```bash
pip install insightface onnxruntime-gpu numpy scikit-learn tqdm pillow opencv-python
```

## 使用方法

### 1. 修改路径配置

编辑 `src/face_cluster.py` 中的路径配置：

```python
BASE_DIR = Path("/mnt/私密文件")
IMAGES_DIR = BASE_DIR / "images"
ORGANIZED_DIR = BASE_DIR / "organized"
FACES_DIR = BASE_DIR / "faces"
METADATA_DIR = BASE_DIR / "metadata"
```

### 2. 运行程序

```bash
# 后台运行
nohup python3 src/face_cluster.py > logs/face_cluster.log 2>&1 &

# 查看进度
tail -f logs/face_cluster.log

# 查看GPU状态
nvidia-smi
```

### 3. 测试模式

修改脚本中的测试配置：

```python
TEST_MODE = True   # 开启测试模式
TEST_LIMIT = 1000  # 测试图片数量
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| FACE_TOLERANCE | 0.6 | 聚类阈值 (欧氏距离) |
| MAX_CLUSTER_SIZE | 30000 | 单次聚类最大人脸数 |
| BATCH_SIZE | 100 | 批处理大小 |

## 输出结构

```
/mnt/私密文件/
├── images/                    # 原始图片
├── faces/
│   ├── face_cache_v3.pkl      # 人脸特征缓存
│   └── face_XXX/              # 旧分组（保留）
├── organized/                 # ⭐ 整理后输出
│   ├── cluster_XXXXX/         # 人脸分组
│   └── no_face/               # 无人脸图片
└── metadata/
    └── clusters.json          # 元数据
```

## 性能数据

| 硬件 | 处理速度 | 173k图片预计时间 |
|------|----------|------------------|
| RTX 3060 6GB | ~40张/秒 | ~1.5小时 |

## 项目结构

```
人脸识别工具/
├── src/
│   └── face_cluster.py        # 主程序
├── docs/
│   ├── README.md              # 说明文档
│   └── CHANGELOG.md           # 更新日志
├── logs/                      # 日志目录
└── config/                    # 配置目录
```

## 维护指南

### 断点续传

程序会自动保存缓存，支持中断后继续处理：

```bash
# 查看缓存
ls -lh /mnt/私密文件/faces/face_cache_v3.pkl
```

### 常见问题

**Q: GPU 未检测到**
A: 检查 CUDA 驱动安装：`nvidia-smi`

**Q: 内存不足**
A: 减小 `MAX_CLUSTER_SIZE` 参数

**Q: 处理速度慢**
A: 检查 GPU 利用率，确保使用 CUDA 版本

## 更新日志

### v3.0 (2026-03-08)
- 基于 InsightFace GPU 加速重构
- 支持断点续传
- 修复聚类参数问题
- 优化内存使用
