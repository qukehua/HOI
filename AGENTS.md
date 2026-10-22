# Project working directory

The canonical project root is `D:/code/HOI`, as requested by the user.
Use this directory as the working directory for all project commands and changes.
The former `C:/Users/qkh/Documents/ChatGPT/HOI 2` tree is an older staging copy.

- Local Python: `.venv/Scripts/python.exe`.
- Existing raw OMOMO data: `OMOMO/data`; preserve these user-provided files.
- `data/` contains only current parametric training/evaluation dependencies and `data/README.md`.
- On 2026-10-21 the user explicitly requested deleting raw BEHAVE, original text annotations, old body-FK datasets/manifests and unused preparation sidecars. Those inputs were deleted; current native data and AE weights are complete. Historical cleanup totals are retained in `docs/cleanup_tests_docs_20261022.json`.
- Default training configuration: `configs/unified.yaml`.
- A second, isolated model is `src/unified_hoi/token_ar/`, with configs in `configs/token_ar/` and outputs in `runs/token_ar/`. See `docs/TOKEN_AR.md`. Preserve the original `parametric/` diffusion path and its public CLIs. Token-AR has copied native body/data/metric helpers but must not import diffusion models, trainers, or contact AE weights. Shared native data are read-only; VQ codebooks and AR checkpoints are dataset-specific and incompatible with diffusion checkpoints.
- The user selected Qwen3.5-4B-Base full text-backbone tuning with shared H/O/C task masks (no LoRA). Use `configs/token_ar/qwen35_4b_{behave,omomo}.yaml` with `accelerate_qwen35_4gpu.yaml`; old Qwen3 configs are optional comparisons. VQ codebooks are pretrained separately and frozen during AR training. CPU smoke checks do not establish 4B generation quality or four-GPU memory fit.
- The original public CLIs dispatch to `src/unified_hoi/parametric`; Token-AR has separate CLIs. The 2026-10-21 legacy cleanup completed. On 2026-10-22 the user requested removing test code and unnecessary documentation again; see `docs/cleanup_tests_docs_20261022.json` for the fixed plan and actual status. Preserve current model source, configs, operational docs and provenance. Only `scripts/cleanup_tests_docs.ps1` is current; do not run superseded cleanup scripts.
- The user explicitly requested keeping the entire `external/` directory, including all reference repositories and their contents. Exclude it and every descendant from cleanup.
- Check training and sampling CLI imports with `.venv/Scripts/python.exe -B -m unified_hoi.train --help` and `.venv/Scripts/python.exe -B -m unified_hoi.sample --help`.
- The research implementation and its current limits are documented in `README.md` and `docs/PARAMETRIC_RETRAINING.md`. Preserve the vendored TriDi primitives and their MIT license.
