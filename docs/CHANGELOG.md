# Changelog

## v3.1 (2026-03-08)

### Fixed
- Replaced the large-scale fallback from staged agglomerative clustering to automatic Birch clustering
- Avoided OOM during the cross-batch merge stage on 155k+ face embeddings
- Normalized embeddings before clustering to keep distance behavior stable across backends
- Added copy fallback when hard-link creation is not permitted on the target filesystem
- Expanded image scanning coverage to include `.jpe`, `.bmp`, `.tif`, and `.tiff`
- Switched file discovery to case-insensitive suffix matching so mixed-case extensions are not skipped

### Added
- Metadata output now records the clustering backend and thresholds used for the run
- Incident report for the 2026-03-08 OOM failure

## v3.0 (2026-03-08)

### Added
- InsightFace (ONNX) GPU extraction
- Resume support through `face_cache_v3.pkl`
- Periodic cache persistence during extraction

### Fixed
- Multi-face image handling
- Filename collision handling during output linking
