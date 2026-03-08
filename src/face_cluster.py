#!/usr/bin/env python3
"""
Offline face clustering tool.

The extraction stage uses InsightFace with ONNXRuntime GPU support when
available. For clustering, the script keeps agglomerative clustering for
smaller datasets and automatically switches to Birch for large datasets to
avoid OOM on pairwise-distance construction.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

try:
    import onnxruntime as ort  # noqa: F401

    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: ONNXRuntime is not installed, falling back to CPU mode.")

BASE_DIR = Path("/mnt/私密文件")
IMAGES_DIR = BASE_DIR / "images"
ORGANIZED_DIR = BASE_DIR / "organized"
FACES_DIR = BASE_DIR / "faces"
METADATA_DIR = BASE_DIR / "metadata"

NUM_WORKERS = 4
FACE_TOLERANCE = 0.6
MAX_AGGLOMERATIVE_SIZE = 30_000
BIRCH_THRESHOLD = 0.75
BIRCH_BRANCHING_FACTOR = 50
TEST_MODE = False
TEST_LIMIT = 1_000


def get_image_files(directory: Path) -> List[Path]:
    exts = [".jpg", ".jpeg", ".png", ".gif", ".webp"]
    files: List[Path] = []
    for ext in exts:
        files.extend(directory.rglob(f"*{ext}"))
        files.extend(directory.rglob(f"*{ext.upper()}"))
    return files


def get_safe_filename(image_path: Path) -> str:
    rel_path = image_path.relative_to(IMAGES_DIR)
    hash_suffix = hashlib.md5(str(rel_path).encode("utf-8")).hexdigest()[:8]
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
                return str(image_path), []

            faces = app.get(img)
            encodings = [face.embedding for face in faces]
            return str(image_path), encodings

        import face_recognition

        image = face_recognition.load_image_file(str(image_path))
        encodings = face_recognition.face_encodings(image)
        return str(image_path), encodings
    except Exception:
        return str(image_path), []


def normalize_embeddings(encodings: Sequence[np.ndarray]) -> np.ndarray:
    matrix = np.asarray(encodings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def cluster_faces(encodings: Sequence[np.ndarray], tolerance: float) -> List[int]:
    if len(encodings) < 2:
        return list(range(len(encodings)))

    if len(encodings) <= MAX_AGGLOMERATIVE_SIZE:
        print(
            f"Using agglomerative clustering for {len(encodings)} faces "
            f"(tolerance={tolerance})."
        )
        return cluster_faces_agglomerative(encodings, tolerance)

    print(
        f"Detected {len(encodings)} faces, which exceeds the safe limit for "
        f"agglomerative clustering ({MAX_AGGLOMERATIVE_SIZE})."
    )
    print(
        "Switching to Birch incremental clustering to avoid pairwise-distance "
        "OOM."
    )
    return cluster_faces_birch(encodings)


def cluster_faces_agglomerative(
    encodings: Sequence[np.ndarray], tolerance: float
) -> List[int]:
    from sklearn.cluster import AgglomerativeClustering

    matrix = normalize_embeddings(encodings)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=tolerance,
        metric="euclidean",
        linkage="ward",
    )
    return clustering.fit_predict(matrix).tolist()


def cluster_faces_birch(encodings: Sequence[np.ndarray]) -> List[int]:
    from sklearn.cluster import Birch

    matrix = normalize_embeddings(encodings)
    print(
        f"Running Birch on matrix {matrix.shape} "
        f"(threshold={BIRCH_THRESHOLD}, branching_factor={BIRCH_BRANCHING_FACTOR})."
    )
    clustering = Birch(
        threshold=BIRCH_THRESHOLD,
        branching_factor=BIRCH_BRANCHING_FACTOR,
        n_clusters=None,
    )
    labels = clustering.fit_predict(matrix)
    sizes = np.bincount(labels)
    print(
        f"Birch finished: {len(sizes)} clusters, "
        f"largest cluster={int(sizes.max())}, "
        f"singletons={int((sizes == 1).sum())}."
    )
    return labels.tolist()


def link_clustered_files(
    valid_paths: Sequence[str], labels: Sequence[int], face_data: Dict[str, List[List[float]]]
) -> Dict[str, int]:
    ORGANIZED_DIR.mkdir(parents=True, exist_ok=True)

    unique_labels = sorted(set(labels))
    for label in unique_labels:
        cluster_dir = ORGANIZED_DIR / f"cluster_{label:05d}"
        cluster_dir.mkdir(exist_ok=True)

    path_to_labels: Dict[str, set[int]] = defaultdict(set)
    for path, label in zip(valid_paths, labels):
        path_to_labels[path].add(label)

    linked_files = 0
    for path, label_set in path_to_labels.items():
        src = Path(path)
        for label in sorted(label_set):
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
                linked_files += 1
            except Exception as exc:
                try:
                    shutil.copy2(src, dst)
                    linked_files += 1
                except Exception as copy_exc:
                    print(
                        f"Failed to materialize {src} -> {dst}: "
                        f"link error={exc}; copy error={copy_exc}"
                    )

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
                except Exception:
                    try:
                        shutil.copy2(src, dst)
                    except Exception:
                        pass
            no_face_count += 1

    return {
        "cluster_count": len(unique_labels),
        "linked_files": linked_files,
        "images_without_faces": no_face_count,
    }


def main() -> None:
    print("=" * 60)
    print("Face clustering tool v4 - resume-safe large-scale mode")
    print("=" * 60)

    print("\n[1/5] Scanning image files...")
    image_files = get_image_files(IMAGES_DIR)
    print(f"Found {len(image_files)} image files.")

    if TEST_MODE:
        image_files = image_files[:TEST_LIMIT]
        print(f"Test mode enabled. Only processing the first {TEST_LIMIT} images.")

    print(f"\n[2/5] Extracting face embeddings (GPU available: {INSIGHTFACE_AVAILABLE})...")
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = FACES_DIR / "face_cache_v3.pkl"

    if cache_file.exists():
        with cache_file.open("rb") as fh:
            face_data: Dict[str, List[List[float]]] = pickle.load(fh)
        print(f"Loaded cache with {len(face_data)} image entries.")
    else:
        face_data = {}

    todo_files = [f for f in image_files if str(f) not in face_data]
    print(f"Remaining files that need extraction: {len(todo_files)}")

    app = None
    if INSIGHTFACE_AVAILABLE and todo_files:
        print("Initializing InsightFace with CUDAExecutionProvider...")
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("GPU acceleration is enabled.")

    if todo_files and app is not None:
        print("Using GPU single-threaded extraction...")
        processed = 0
        with tqdm(total=len(todo_files), desc="face-detect") as pbar:
            for image_file in todo_files:
                path, encodings = extract_all_face_embeddings(image_file, app)
                face_data[path] = [
                    embedding.tolist() if embedding is not None else None
                    for embedding in encodings
                ]
                processed += 1
                pbar.update(1)

                if processed % 500 == 0:
                    with cache_file.open("wb") as fh:
                        pickle.dump(face_data, fh)

        with cache_file.open("wb") as fh:
            pickle.dump(face_data, fh)
    elif todo_files:
        print("Using CPU multi-threaded extraction...")
        processed = 0
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(extract_all_face_embeddings, image_file, app): image_file
                for image_file in todo_files
            }

            with tqdm(total=len(todo_files), desc="face-detect") as pbar:
                try:
                    for future in as_completed(futures):
                        path, encodings = future.result()
                        face_data[path] = [
                            embedding.tolist() if embedding is not None else None
                            for embedding in encodings
                        ]
                        processed += 1
                        pbar.update(1)

                        if processed % 500 == 0:
                            with cache_file.open("wb") as fh:
                                pickle.dump(face_data, fh)
                finally:
                    with cache_file.open("wb") as fh:
                        pickle.dump(face_data, fh)

    print("\n[3/5] Preparing clustering inputs...")
    valid_encodings: List[np.ndarray] = []
    valid_paths: List[str] = []

    for path, encodings in face_data.items():
        for encoding in encodings:
            if encoding is not None and len(encoding) > 0:
                valid_encodings.append(np.asarray(encoding, dtype=np.float32))
                valid_paths.append(path)

    print(
        f"Detected {len(valid_encodings)} face embeddings across "
        f"{len(set(valid_paths))} images."
    )

    print("\n[4/5] Clustering faces...")
    labels = cluster_faces(valid_encodings, FACE_TOLERANCE)
    unique_labels = set(labels)
    print(f"Cluster count: {len(unique_labels)}")

    print("\n[5/5] Linking organized output...")
    stats = link_clustered_files(valid_paths, labels, face_data)
    print(f"Images without faces: {stats['images_without_faces']}")
    print(f"Linked files: {stats['linked_files']}")

    metadata = {
        "total_images": len(image_files),
        "total_faces": len(valid_encodings),
        "images_with_faces": len(set(valid_paths)),
        "images_without_faces": stats["images_without_faces"],
        "cluster_count": stats["cluster_count"],
        "cluster_backend": (
            "agglomerative"
            if len(valid_encodings) <= MAX_AGGLOMERATIVE_SIZE
            else "birch"
        ),
        "face_tolerance": FACE_TOLERANCE,
        "max_agglomerative_size": MAX_AGGLOMERATIVE_SIZE,
        "birch_threshold": BIRCH_THRESHOLD,
        "birch_branching_factor": BIRCH_BRANCHING_FACTOR,
        "cache_file": str(cache_file),
    }

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with (METADATA_DIR / "clusters.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Output directory: {ORGANIZED_DIR}")
    print(f"Metadata file: {METADATA_DIR / 'clusters.json'}")


if __name__ == "__main__":
    main()
