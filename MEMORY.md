# CryoDDM GUI Memory

## Long-Term Context
- 2026-07-07: The current priority is GUI responsiveness and stability. Reported symptoms: the app is generally sluggish, can crash, window dragging/resizing stutters, resize/layout changes can lag and then distort the layout, and image loading must handle cryo-EM files around 270 MB each with up to 50 selected images.
- 2026-07-07: The project did not include project-local `AGENTS.md` or `MEMORY.md`; these were created from detected project facts because no template directory was available.

## Design Constraints
- Large MRC/MRCS images should be loaded lazily and rendered from display-sized data where possible. Keeping 50 full-resolution processed images in memory is expected to be unsafe on normal workstations.
- GUI-thread blocking work is a primary risk area. MRC I/O, normalization, adaptive histogram equalization, and pixmap creation should be scheduled carefully to keep interactions responsive.
- 2026-07-07: GUI image browsing now uses bounded preview loading (`MAX_PREVIEW_EDGE=4096`), small neighbor prefetch, and an LRU pixmap cache. Manual annotations are stored in original MRC pixel coordinates through preview-to-source coordinate scaling.
- 2026-07-07: Custom frame performance favors responsiveness over decorative effects: translucent background and drop shadow are disabled, resize grips use global drag deltas, and the old forced `500x500` right-grip resize was removed.
- 2026-07-10: The project now has pip/PyPI packaging metadata in `pyproject.toml`. The `cryoddm` console command executes the existing `main.py` path through a small wrapper instead of refactoring the current flat source layout.
- 2026-07-10: `environment.yaml` no longer sets a pip-wide PyTorch CUDA index. Conda handles PyTorch/CUDA and scientific GUI dependencies; pip is only used for pyem from GitHub.
- 2026-07-10: User-facing startup is `cryoddm` after pip installation. The old source-file launch style should not be presented as the normal launch path.
