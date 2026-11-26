# Token-AR for Human–Object Interaction

Token-AR encodes human motion, object motion, and contact with three separate residual VQ-VAEs, then generates the missing modalities with a fully fine-tuned [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base). Training, codebooks, and checkpoints are kept separate for each dataset. All commands run from the repository root.

## Tasks

One backbone per dataset covers all three tasks. BEHAVE uses SMPL-H; OMOMO and GRAB use SMPL-X.

| Dataset | Body | Text-to-interaction | Object-to-human | Human-to-object |
| --- | --- | --- | --- | --- |
| BEHAVE | SMPL-H | ✔ | ✔ | ✔ |
| OMOMO | SMPL-X | ✔ | ✔ | ✔ |
| GRAB | SMPL-X | ✔ | ✔ | ✔ |

VQ-VAE and motion–text evaluator training support all three datasets. Qwen backbone configs are provided for BEHAVE and OMOMO; GRAB follows the same recipe once its codebook is ready.

## Installation

Training, evaluation, and inference use the `tokenar` environment.

```bash
conda create -n tokenar python=3.12 -y
conda activate tokenar
pip install -r requirements.txt
```

Place the body models under `data/smplx_models/`. SMPL-H is the Extended SMPL+H model from the [MANO download page](https://mano.is.tue.mpg.de/download.php). SMPL-X is from the [SMPL-X download page](https://smpl-x.is.tue.mpg.de/download.php).

```text
data/smplx_models
├── smplh
│   ├── SMPLH_FEMALE.pkl
│   └── SMPLH_MALE.pkl
└── smplx
    ├── SMPLX_FEMALE.npz
    └── SMPLX_MALE.npz
```

## Data

Canonical releases follow the [InterAct](https://github.com/wzyabcas/InterAct) layout: `human.npz` and `object.npz` per sequence. Write one JSONL manifest per dataset. Each line points `path` at a sequence directory (before conversion) or a parametric `.npz` (after conversion) and sets `split` to `train`, `val`, or `test`.

```json
{"path": "data/datasets/behave_correct/sequences_canonical/Date01_Sub01_backpack_back_0", "dataset": "behave", "sequence_id": "Date01_Sub01_backpack_back_0", "split": "train", "fps": 30}
```

VQ, evaluator, and backbone training use the `*_correct_evaluator.jsonl` manifests, which add CLIP text features under `data/parametric_assets/<dataset>/clip_vit_b32/`.

| Dataset | Sequences | Object mesh | Contact support | Manifest |
| --- | --- | --- | --- | --- |
| BEHAVE | `data/datasets/behave_correct/sequences_canonical/<id>/` | `data/datasets/behave_raw/objects/<name>/<name>.obj` | `data/parametric_assets/behave/contact_support.npz` | `data/processed/behave_correct_evaluator.jsonl` |
| OMOMO | `data/datasets/omomo_correct/sequences_canonical/<id>/` | `data/datasets/omomo_correct/objects/<name>/<name>.obj` | `data/parametric_assets/omomo/contact_support.npz` | `data/processed/omomo_correct_evaluator.jsonl` |
| GRAB | `data/datasets/grab_correct/sequences_canonical/<id>/` | `data/datasets/grab_correct/objects/<name>/<name>.obj` | `data/parametric_assets/grab/contact_support.npz` | `data/processed/grab_correct_evaluator.jsonl` |

The BEHAVE corrected release contains poses only; its meshes stay with the original BEHAVE object scans. OMOMO and GRAB corrected releases already include `objects/`. GRAB hand pose is stored as SMPL-X PCA expanded to full articulation (`hand_source=grab_pca24_expanded`).

## Preprocessing

Convert each InterAct sequence into the parametric records used for training. The script rewrites `path` in the manifest to the exported `.npz` and writes `contact_support.npz` under `data/parametric_assets/<dataset>/`.

```bash
conda activate tokenar
python -m tokenar_hoi.token_ar.interact_convert \
  --manifest data/processed/behave_correct.jsonl \
  --output data/parametric_assets/behave/sequences \
  --workers 16 \
  --body-models data/smplx_models
```

For OMOMO, replace the manifest with `data/processed/omomo_correct.jsonl` and the output with `data/parametric_assets/omomo/sequences`. For GRAB, use `data/processed/grab_correct.jsonl` and `data/parametric_assets/grab/sequences`.

## Training

**Motion–text evaluator.** Train before VQ selection-by-FID. Configs: `config/evaluator_behave.yaml`, `config/evaluator_omomo.yaml`, `config/evaluator_grab.yaml`.

```bash
conda activate tokenar
python -m tokenar_hoi.token_ar.train_evaluator --config config/evaluator_behave.yaml
```

Checkpoints land at `runs/token_ar/<dataset>/motion_text_evaluator/last.pt`.

**VQ-VAE.** Each dataset uses a single-GPU residual EMA codebook (`gpus: 1`, `batch_size: 512`, `max_steps: 150000`). Human / object / contact train for 150k / 100k / 100k steps. `best.pt` is the lowest validation joint FID against the frozen evaluator.

```bash
conda activate tokenar
python -m tokenar_hoi.token_ar.train_codecs --config config/vq_behave.yaml
python -m tokenar_hoi.token_ar.train_codecs --config config/vq_omomo.yaml
python -m tokenar_hoi.token_ar.train_codecs --config config/vq_grab.yaml
```

Checkpoints are `runs/token_ar/<dataset>/vq_correct/{last,best}.pt`. The output directory must be empty on a fresh run. Resume with `--resume` and the same `last.pt`.

**Backbone.** In `config/qwen35_4b_behave.yaml` and `config/qwen35_4b_omomo.yaml`, set `manifest` to the corresponding `*_correct_evaluator.jsonl`, `codecs` to `runs/token_ar/<dataset>/vq_correct/best.pt`, and `output` to a new directory (for example `runs/token_ar/<dataset>/qwen35_4b_tasks3`). Training is full-parameter on 4 GPUs.

```bash
conda activate tokenar
accelerate launch --config_file config/accelerate_qwen35_4gpu.yaml \
  --module tokenar_hoi.token_ar.train --config config/qwen35_4b_behave.yaml
```

Swap the config for OMOMO. Resume with `--resume runs/token_ar/<dataset>/<run>/checkpoint-XXXXXXX` and the same process count. Modes `111` / `101` / `011` are trained jointly; `text_dropout: 0.5` applies only to object-to-human.

The size ablation uses the same tasks, codebooks, and hyperparameters with Qwen3.5-9B-Base (`config/qwen35_9b_behave.yaml`, `config/qwen35_9b_omomo.yaml`). Qwen3.5 has no 8B release. Launch those configs with the same accelerate file. Both pretrained backbones are read from `data/pretrained/Qwen3.5-4B-Base` and `data/pretrained/Qwen3.5-9B-Base`.

## Evaluation

```bash
conda activate tokenar
python -m tokenar_hoi.token_ar.evaluate_codecs \
  --checkpoint runs/token_ar/behave/vq_correct/best.pt \
  --evaluator runs/token_ar/behave/motion_text_evaluator/last.pt \
  --split val --device cuda:0 \
  --output runs/eval_behave_vq

python -m tokenar_hoi.token_ar.benchmark \
  --checkpoint runs/token_ar/behave/qwen35_4b_tasks3/checkpoint-XXXXXXX \
  --motion-text-evaluator runs/token_ar/behave/motion_text_evaluator/last.pt \
  --split test --temperature 0 \
  --device cuda:0 --dtype bfloat16 \
  --output runs/eval_behave_backbone
```

`benchmark` reports text-to-HOI, object-to-human, and human-to-object. Replace `behave` with `omomo` or `grab` for the other datasets once the corresponding backbone checkpoint exists. `--output` must be a new directory. Metrics are written to `summary.json`.

## Visualization

```bash
conda activate tokenar
python -m tokenar_hoi.token_ar.sample \
  --checkpoint runs/token_ar/behave/qwen35_4b_tasks3/checkpoint-XXXXXXX \
  --manifest data/processed/behave_correct_evaluator.jsonl \
  --reference data/parametric_assets/behave/sequences/Date01_Sub01_backpack_back_0.npz \
  --task text_to_hoi --frames 120 \
  --device cuda:0 --dtype bfloat16 \
  --output runs/vis/behave_sample.npz

python -m tokenar_hoi.smpl_mesh \
  --input runs/vis/behave_sample.npz \
  --output runs/vis/behave_sample_mesh.npz
```

`--task` is `text_to_hoi`, `object_to_human`, or `human_to_object`. The mesh archive stores per-frame `vertices` and `faces`.

## Refinement

After an AR checkpoint is available, cache frozen rollouts and train the geometry refiner. The refiner YAML must set `kind: token_ar_refiner_v1`, `cache`, and `output`.

```bash
conda activate tokenar
python -m tokenar_hoi.token_ar.prepare_refinement \
  --checkpoint runs/token_ar/behave/qwen35_4b_tasks3/checkpoint-XXXXXXX \
  --output runs/token_ar/behave/refinement_cache \
  --device cuda:0 --dtype bfloat16

python -m tokenar_hoi.token_ar.train_refiner --config path/to/refiner.yaml
```

`benchmark` and `sample` accept an optional `--refiner` checkpoint. OMOMO freezes hand-pose residuals in the refiner; BEHAVE and GRAB do not.
