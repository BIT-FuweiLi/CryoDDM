# CryoDDM GUI Performance Tech Spec

## Current Root Causes
- `MRCFullLoadRunnable` reads and processes full-resolution image arrays, then emits full-size `QImage` objects. A 270 MB input can create several temporary arrays per worker.
- `PREVIEW_CACHE` evicts entries, but `image_list` keeps strong `QPixmap` references, so memory is not actually released.
- Small and medium selections can preload all files, which is unsafe for large MRC data.
- Display coordinates are treated as original coordinates, which prevents simple preview downsampling unless coordinate mapping is added.
- Custom resize grips use local mouse deltas while changing window geometry, causing jitter.
- `Widgets.right()` resizes the whole parent window to `500x500` when constructing the right grip.
- Translucent frameless window background plus drop shadow increases move/resize repaint cost.

## Implementation Plan
- Replace full-resolution display loading with bounded preview loading:
  - Use memory-mapped MRC access for uncompressed files when available.
  - Downsample to `MAX_PREVIEW_EDGE` before contrast normalization.
  - Emit preview `QImage` plus metadata containing original shape, displayed shape, and coordinate scale.
- Switch to lazy loading for all batches:
  - Load current image and a small neighbor window.
  - Keep a bounded LRU preview cache.
  - On eviction, clear `image_list[index]` and `loaded_indices`.
- Preserve original-coordinate annotation:
  - Convert scene preview coordinates to source coordinates on click.
  - Convert saved source coordinates back to preview coordinates when drawing boxes.
  - Validate bounds against original image dimensions.
- Improve graphics view performance:
  - Use `QGraphicsView` optimization flags and bounded viewport updates.
  - Avoid pixmap item device-coordinate caching for large changing images.
- Improve custom frame interactions:
  - Remove forced parent resize in right grip creation.
  - Use global mouse positions for edge resize delta calculation.
  - Disable expensive translucent background and drop shadow by default.

## Verification
- Run `python -m py_compile` for edited Python files.
- Run targeted import/smoke checks if PySide6 is available.
- If real 270 MB MRC files are unavailable, validate logic with static review and synthetic MRC data where dependencies allow it.

## Packaging
- The project can be installed as the `cryoddm` Python distribution.
- `pyproject.toml` is the packaging source of truth for pip installs.
- The `cryoddm` console command is the supported application launcher.
- The legacy Conda environment remains available for servers that need explicit CUDA/PyTorch solver control.
- `cs2star` requires the external `csparc2star.py` command from pyem to be available on `PATH`.
