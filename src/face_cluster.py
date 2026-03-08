#!/usr/bin/env python3
"""
face_cluster.py - 纯离线人脸聚类整理工具 v3
基于方案.md: 使用 InsightFace (ONNX) GPU 加速
"""

import os
import pickle
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

try:
    import onnxruntime as ort

    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: ONNX not available, falling back to CPU")

BASE_DIR = Path("/mnt/私密文件")
IMAGES_DIR = BASE_DIR / "images"
ORGANIZED_DIR = BASE_DIR / "organized"
FACES_DIR = BASE_DIR / "faces"
METADATA_DIR = BASE_DIR / "metadata"

NUM_WORKERS = 4
BATCH_SIZE = 100
FACE_TOLERANCE = 0.6
MAX_CLUSTER_SIZE = 30000
TEST_MODE = False  # 全量处理
TEST_LIMIT = 1000


def get_image_files(directory: Path) -> List[Path]:
    exts = [".jpg", ".jpeg", ".png", ".gif", ".webp"]
    files = []
    for ext in exts:
        files.extend(directory.rglob(f"*{ext}"))
        files.extend(directory.rglob(f"*{ext.upper()}"))
    return files


def get_safe_filename(image_path: Path) -> str:
    rel_path = image_path.relative_to(IMAGES_DIR)
    hash_suffix = hashlib.md5(str(rel_path).encode()).hexdigest()[:8]
    stem = image_path.stem[:50]
    return f"{stem}_{hash_suffix}{image_path.suffix}"


def extract_all_face_embeddings(
    image_path: Path, app=None
) -> Tuple[str, List[np.ndarray]]:
    try:
        if app is not None:
            import cv2

            img = cv2.imread(str(image_path))
            if img is None:
                return (str(image_path), [])
            faces = app.get(img)
            encodings = [face.embedding for face in faces]
            return (str(image_path), encodings)

        import face_recognition

        image = face_recognition.load_image_file(str(image_path))
        encodings = face_recognition.face_encodings(image)
        return (str(image_path), encodings)
    except Exception as e:
        pass

    return (str(image_path), [])


def batch_cluster_faces(
    encodings: List[np.ndarray],
    labels: List[str],
    tolerance: float = 0.6,
    max_batch: int = 30000,
) -> List[int]:
    if len(encodings) <= max_batch:
        return _cluster_single(encodings, tolerance)

    print(f"人脸数量 {len(encodings)} 超过限制 {max_batch}，使用两阶段聚类...")

    batch_count = (len(encodings) + max_batch - 1) // max_batch

    print("阶段 1: 提取每批代表向量...")
    batch_representatives = []
    all_batch_labels = []

    for i in range(batch_count):
        start = i * max_batch
        end = min((i + 1) * max_batch, len(encodings))

        batch_encodings = encodings[start:end]
        batch_labels = _cluster_single(batch_encodings, tolerance)
        all_batch_labels.append(batch_labels)

        unique_batch_labels = set(batch_labels)
        for cluster_id in unique_batch_labels:
            cluster_indices = [j for j, l in enumerate(batch_labels) if l == cluster_id]
            cluster_encodings = [batch_encodings[j] for j in cluster_indices]
            center = np.mean(cluster_encodings, axis=0)
            batch_representatives.append((i, cluster_id, center))

        print(
            f"批次 {i + 1}/{batch_count} 完成，提取 {len(unique_batch_labels)} 个代表"
        )

    print("阶段 2: 跨批合并...")
    representative_encodings = [r[2] for r in batch_representatives]
    merged_labels = _cluster_single(representative_encodings, tolerance)

    print("阶段 3: 映射回原数据...")

    rep_to_merged = {}
    for idx, (batch_idx, cluster_id, _) in enumerate(batch_representatives):
        rep_to_merged[(batch_idx, cluster_id)] = merged_labels[idx]

    final_labels = [0] * len(encodings)

    for i in range(batch_count):
        batch_labels = all_batch_labels[i]

        start = i * max_batch
        end = min((i + 1) * max_batch, len(encodings))

        for j, orig_idx in enumerate(range(start, end)):
            batch_cluster_id = batch_labels[j]
            final_labels[orig_idx] = rep_to_merged[(i, batch_cluster_id)]

    print(f"两阶段聚类完成，最终 {len(set(final_labels))} 个簇")
    return final_labels


def _cluster_single(encodings: List[np.ndarray], tolerance: float = 0.6) -> List[int]:
    if len(encodings) < 2:
        return list(range(len(encodings)))

    from sklearn.cluster import AgglomerativeClustering

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=tolerance,
        metric="euclidean",
        linkage="ward",
    )

    labels = clustering.fit_predict(encodings)
    return labels.tolist()


def main():
    print("=" * 50)
    print("人脸聚类整理工具 v3 - GPU 加速版")
    print("=" * 50)

    print("\n[1/5] 扫描图片文件...")
    image_files = get_image_files(IMAGES_DIR)
    print(f"找到 {len(image_files)} 个图片文件")

    if TEST_MODE:
        image_files = image_files[:TEST_LIMIT]
        print(f"⚠️ 测试模式: 仅处理前 {TEST_LIMIT} 张图片")

    print(f"\n[2/5] 提取人脸特征 (GPU: {INSIGHTFACE_AVAILABLE})...")

    FACES_DIR.mkdir(parents=True, exist_ok=True)

    cache_file = FACES_DIR / "face_cache_v3.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            face_data = pickle.load(f)
        print(f"加载已有缓存: {len(face_data)} 条")
    else:
        face_data = {}

    todo_files = [f for f in image_files if str(f) not in face_data]
    print(f"需要处理: {len(todo_files)} 个文件")

    app = None
    if INSIGHTFACE_AVAILABLE:
        print("初始化 InsightFace GPU 加速...")
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("GPU 加速已启用")

    if app is not None:
        print("使用 GPU 单线程处理...")
        processed = 0
        remaining_files = [f for f in todo_files if str(f) not in face_data]

        with tqdm(total=len(todo_files), desc="人脸检测") as pbar:
            for f in remaining_files:
                path, encodings = extract_all_face_embeddings(f, app)
                face_data[path] = [
                    e.tolist() if e is not None else None for e in encodings
                ]
                processed += 1
                pbar.update(1)

                if processed % 500 == 0:
                    with open(cache_file, "wb") as f:
                        pickle.dump(face_data, f)

        with open(cache_file, "wb") as f:
            pickle.dump(face_data, f)
    else:
        print("使用 CPU 多线程处理...")
        processed = 0
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(extract_all_face_embeddings, f, app): f
                for f in todo_files
            }

            with tqdm(total=len(todo_files), desc="人脸检测") as pbar:
                try:
                    for future in as_completed(futures):
                        path, encodings = future.result()
                        face_data[path] = [
                            e.tolist() if e is not None else None for e in encodings
                        ]
                        processed += 1
                        pbar.update(1)

                        if processed % 500 == 0:
                            with open(cache_file, "wb") as f:
                                pickle.dump(face_data, f)
                finally:
                    with open(cache_file, "wb") as f:
                        pickle.dump(face_data, f)

    print("\n[3/5] 准备聚类数据...")

    valid_encodings = []
    valid_paths = []

    for path, encodings in face_data.items():
        for encoding in encodings:
            if encoding is not None and len(encoding) > 0:
                valid_encodings.append(np.array(encoding))
                valid_paths.append(path)

    print(f"检测到 {len(valid_encodings)} 张人脸 (含多人脸)")

    print("\n[4/5] 人脸聚类...")

    if valid_encodings:
        labels = batch_cluster_faces(
            valid_encodings,
            valid_paths,
            tolerance=FACE_TOLERANCE,
            max_batch=MAX_CLUSTER_SIZE,
        )

        unique_labels = set(labels)
        print(f"聚类数量: {len(unique_labels)}")

        print("\n[5/5] 整理文件...")

        ORGANIZED_DIR.mkdir(parents=True, exist_ok=True)

        for label in unique_labels:
            cluster_dir = ORGANIZED_DIR / f"cluster_{label:05d}"
            cluster_dir.mkdir(exist_ok=True)

        from collections import defaultdict

        path_to_labels = defaultdict(set)
        for path, label in zip(valid_paths, labels):
            path_to_labels[path].add(label)

        for path, label_set in path_to_labels.items():
            src = Path(path)
            for label in label_set:
                safe_name = get_safe_filename(src)
                base_name = safe_name
                counter = 1
                while True:
                    cluster_dir = ORGANIZED_DIR / f"cluster_{label:05d}"
                    dst = cluster_dir / safe_name
                    if not dst.exists():
                        break
                    safe_name = (
                        f"{Path(base_name).stem}_{counter}{Path(base_name).suffix}"
                    )
                    counter += 1

                try:
                    os.link(src, dst)
                except Exception as e:
                    print(f"链接失败: {e}")

        no_face_dir = ORGANIZED_DIR / "no_face"
        no_face_dir.mkdir(exist_ok=True)

        no_face_count = 0
        for path, encodings in face_data.items():
            if not encodings or all(e is None for e in encodings):
                src = Path(path)
                safe_name = get_safe_filename(src)
                dst = no_face_dir / safe_name
                if not dst.exists():
                    try:
                        os.link(src, dst)
                    except:
                        pass
                no_face_count += 1

        print(f"无人脸图片: {no_face_count} 张")

        metadata = {
            "total_images": len(image_files),
            "total_faces": len(valid_encodings),
            "images_with_faces": len(set(valid_paths)),
            "images_without_faces": no_face_count,
            "cluster_count": len(unique_labels),
        }

        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(METADATA_DIR / "clusters.json", "w") as f:
            json.dump(metadata, f, indent=2)

    print("\n✅ 处理完成!")
    print(f"输出目录: {ORGANIZED_DIR}")
    print(f"缓存文件: {cache_file}")


if __name__ == "__main__":
    main()
