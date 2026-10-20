# Project working directory

The canonical project root is `D:/code/HOI`, as requested by the user.
Use this directory as the working directory for all project commands and changes.
The former `C:/Users/qkh/Documents/ChatGPT/HOI 2` tree is an older staging copy.

- Local Python: `.venv/Scripts/python.exe`.
- Existing raw OMOMO data: `OMOMO/data`; preserve these user-provided files.
- Downloaded BEHAVE data: `data/raw/behave`.
- Default training configuration: `configs/unified.yaml`.
- Test sources and smoke artifacts are included in the user-requested cleanup; see docs/project_cleanup_report.json for execution status. Check the training and sampling CLI imports with `.venv/Scripts/python.exe -B -m unified_hoi.train --help` and `.venv/Scripts/python.exe -B -m unified_hoi.sample --help`.
- The research implementation and its current limits are documented in `README.md` and `docs/IMPLEMENTATION_STATUS.md`.
