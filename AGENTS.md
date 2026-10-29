# Project working directory

The current project root on the training server is `/share/personal/qukehua/HOI` (Python: `.venv/bin/python`).
The older Windows copy was `D:/code/HOI` (`.venv/Scripts/python.exe`).

- Existing raw OMOMO data: `OMOMO/data`; preserve these user-provided files.
- `data/` contains only native parametric data, meshes, contact supports and body models. Do not modify it during code cleanup.
- The only model is `src/unified_hoi/token_ar/`, with configs in `configs/token_ar/`. See `docs/TOKEN_AR.md`. On 2026-10-29 the user requested deleting code not needed by token_ar; the diffusion path (`parametric/`, `integrations/`, old CLIs, configs, scripts and docs) was removed and is recoverable from git history. The tridi_metrics_v2 generation metrics (1-NNA/COV/MMD/SD) now live in `token_ar/metrics.py`, with the batch entry `python -m unified_hoi.token_ar.benchmark`.
- The user selected Qwen3.5-4B-Base full text-backbone tuning with shared H/O/C task masks (no LoRA). Use `configs/token_ar/qwen35_4b_{behave,omomo}.yaml` with `accelerate_qwen35_4gpu.yaml`; Qwen3 configs are kept as comparisons. VQ codebooks are pretrained separately and frozen during AR training. On 2026-10-29 the user selected the final VQ checkpoint for new BEHAVE runs: `runs/token_ar/behave/vq/last.pt` (used by `qwen35_4b_behave.yaml`). Changed codebooks require a new AR run; evaluate existing AR runs with their bundled `codecs.pt`.
- The user explicitly requested keeping the entire `external/` directory, including all reference repositories and their contents. Exclude it and every descendant from cleanup.
- Check CLI imports with `python -B -m unified_hoi.token_ar.{train_codecs,train,sample,evaluate,benchmark} --help`.
