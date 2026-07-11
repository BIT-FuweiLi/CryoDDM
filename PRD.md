# CryoDDM GUI Performance PRD

## Problem
The GUI becomes sluggish and may crash when users load large cryo-EM MRC files or move/resize the main window. Typical input can be around 270 MB per image, with up to 50 selected images.

## Users
Cryo-EM users who need to inspect micrographs, mark noise coordinates, and run the CryoDDM pipeline without the GUI freezing.

## Goals
- Keep the GUI responsive while selecting and navigating large MRC batches.
- Support selecting 50 large MRC files without eagerly decoding all files into full-resolution pixmaps.
- Keep manual coordinate picking stored in original image coordinates even when the displayed image is downsampled.
- Make window move and resize interactions stable and smooth enough for normal desktop use.

## Non-Goals
- Change the training, prediction, forward simulation, or cs2star scientific algorithms.
- Add new dependencies or redesign the whole Qt Designer layout.
- Implement a tiled full-resolution image viewer.

## Acceptance Criteria
- Loading a batch starts with the current image and nearby images only, not every selected file.
- The in-memory preview cache has a fixed upper bound and evicts old pixmaps from all strong references.
- Large images are displayed from bounded-size previews.
- Clicking on a preview records coordinates mapped back to the original MRC pixel space.
- Resizing no longer forces the main window back to an arbitrary 500x500 geometry.
- Python files edited for the fix pass `python -m py_compile`.

