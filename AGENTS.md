# Project working directory

The canonical project root is `D:/code/HOI`, as requested by the user.
Use this directory as the working directory for all project commands and changes.
The former `C:/Users/qkh/Documents/ChatGPT/HOI 2` tree is an older staging copy.

- Local Python: `.venv/Scripts/python.exe`.
- Existing raw OMOMO data: `OMOMO/data`; preserve these user-provided files.
- `data/` contains only current parametric training/evaluation dependencies and `data/README.md`.
- On 2026-10-21 the user explicitly requested deleting raw BEHAVE, original text annotations, old body-FK datasets/manifests and unused preparation sidecars. Do not assume those inputs still exist; see `docs/data_cleanup_20261021.json`. Current native data and AE weights are complete.
- Default training configuration: `configs/unified.yaml`.
- The active representation is `parametric_v1`; public CLIs dispatch directly to `src/unified_hoi/parametric`. Legacy code/configs, source conversion tools, tests and verification outputs are scheduled for the user-requested cleanup; see `docs/code_cleanup_20261021.json` for actual execution status. Do not run the superseded cleanup_project.ps1 or cleanup_processed.ps1 scripts.
- The user explicitly requested keeping the entire `external/` directory, including all reference repositories and their contents. Exclude it and every descendant from cleanup.
- Check training and sampling CLI imports with `.venv/Scripts/python.exe -B -m unified_hoi.train --help` and `.venv/Scripts/python.exe -B -m unified_hoi.sample --help`.
- The research implementation and its current limits are documented in `README.md` and `docs/PARAMETRIC_RETRAINING.md`. Preserve the vendored TriDi primitives and their MIT license.
