# CryoDDM GUI Project Rules

## Project Snapshot
- Project name: CryoDDM GUI.
- Workspace date: 2026-07-07.
- Stack: Python 3.10, PySide6 6.4.2, NumPy, scikit-image, mrcfile, Matplotlib, PyTorch.
- Environment: Conda environment defined by `environment.yaml`, named `cryoddm`.
- Entry point: `main.py`.
- Main directories:
  - `modules/`: generated and custom UI/application helpers.
  - `widgets/`: custom Qt widgets and resize grips.
  - `core/`: cryo-EM data processing, model training, prediction, and conversion scripts.
  - `themes/`: Qt stylesheet files.
  - `images/`: app icons and images.

## Commands
- Create environment: `conda env create -f environment.yaml`.
- Activate environment: `conda activate cryoddm`.
- Install local package in the environment: `pip install --no-deps .`.
- Run app: `cryoddm`.
- Prefer focused smoke checks over broad rewrites when validating GUI changes.

## Coding Rules
- Keep changes scoped to the performance, responsiveness, and crash issues being fixed.
- Preserve existing user workflows and UI labels unless a direct bug requires changing them.
- Avoid new dependencies unless the existing stack cannot solve the problem.
- Use Qt threading/signals for long-running image work; do not block the GUI thread with MRC reading, normalization, or expensive rendering.
- Keep image memory bounded. Full-resolution cryo-EM files can be hundreds of MB each, so do not keep unnecessary full-resolution copies for all loaded files.
- Do not hardcode user-specific paths, credentials, GPU IDs, or secrets.

## Verification Expectations
- At minimum, run a Python syntax/import-oriented check for edited Python files.
- For GUI behavior, run `cryoddm` when the local environment supports PySide6.
- For image-loading changes, verify behavior with synthetic or small MRC data when large real data is unavailable.
