# Token-AR for Human–Object Interaction

Token-AR encodes human motion, object motion, and contact with three separate VQ-VAEs, then generates the missing modalities with a fully fine-tuned [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base). Training, codebooks, and checkpoints are kept separate for each dataset. All commands run from the repository root.

## Tasks

One backbone per dataset covers all three tasks. GRAB is not accepted by the VQ-VAE or the backbone.

| Dataset | Text-to-interaction | Object-to-human | Human-to-object |
| --- | --- | --- | --- |
| BEHAVE | ✔ | ✔ | ✔ |
| OMOMO | ✔ | ✔ | ✔ |
| GRAB | — | — | — |

## Installation

```bash
conda create -n hoi python=3.12 -y
conda activate hoi
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

Downloads: [BEHAVE]() · [OMOMO]() · [GRAB]()

Sequences follow the [InterAct](https://github.com/wzyabcas/InterAct) canonical layout: `human.npz` and `object.npz` per sequence. Write one JSONL manifest per dataset. Each line points `path` at a sequence directory and sets `split` to `train`, `val`, or `test`.

```json
{"path": "data/datasets/behave_correct/sequences_canonical/Date01_Sub01_backpack_back_0", "dataset": "behave", "sequence_id": "Date01_Sub01_backpack_back_0", "split": "train", "fps": 30}
```

| Dataset | Sequences | Object mesh | Contact support | Manifest |
| --- | --- | --- | --- | --- |
| BEHAVE | `data/datasets/behave_correct/sequences_canonical/<id>/` | `data/datasets/behave_raw/objects/<name>/<name>.obj` | `data/parametric_assets/behave/contact_support.npz` | `data/processed/behave_correct.jsonl` |
| OMOMO | `data/datasets/omomo_correct/sequences_canonical/<id>/` | `data/datasets/omomo_correct/objects/<name>/<name>.obj` | `data/parametric_assets/omomo/contact_support.npz` | `data/processed/omomo_correct.jsonl` |
| GRAB | `external/InterAct/data/grab/sequences_canonical/<id>/` | `external/InterAct/data/grab/objects/<name>/<name>.obj` | — | — |

The BEHAVE corrected release contains poses only. Its meshes stay with the original BEHAVE object scans. The OMOMO corrected release already includes `objects/`. GRAB is prepared and rendered with InterAct; the VQ-VAE and backbone currently accept `behave` and `omomo`.

## Preprocessing

Convert each InterAct sequence into the parametric records used for training. The script rewrites `path` in the manifest to the exported `.npz`.

```bash
python -m tokenar_hoi.token_ar.interact_convert \
  --manifest data/processed/behave_correct.jsonl \
  --output data/parametric_assets/behave/sequences \
  --workers 16 \
  --body-models data/smplx_models
```

For OMOMO, replace the manifest with `data/processed/omomo_correct.jsonl` and the output with `data/parametric_assets/omomo/sequences`.

To render GRAB, point the visualizer at the canonical sequences and run it inside InterAct:

```bash
cd external/InterAct
ln -sfn sequences data/grab/sequences_canonical
python visualization/visualize.py grab
```

## Training

**VQ-VAE.** BEHAVE reads `config/vq_behave_correct.yaml` (4 GPUs, global batch 2048). Set `gpus` to the number of visible devices and keep `batch_size` divisible by that number. OMOMO uses `config/vq_omomo.yaml`; set `manifest`, `contact_support`, and `output` to the corrected paths in the table above and to `runs/token_ar/omomo/vq_correct`.

```bash
python -m tokenar_hoi.token_ar.train_codecs --config config/vq_behave_correct.yaml
python -m tokenar_hoi.token_ar.train_codecs --config config/vq_omomo.yaml
```

Checkpoints are `runs/token_ar/behave/vq_correct/last.pt` and `runs/token_ar/omomo/vq_correct/last.pt`. The output directory must be empty. Resume with `--resume` and the same `last.pt`.

**Backbone.** In `config/qwen35_4b_behave.yaml` and `config/qwen35_4b_omomo.yaml`, point `manifest` and `contact_support` at the corrected files, `codecs` at the corresponding `last.pt`, and `output` at `runs/token_ar/<dataset>/qwen35_4b_correct`. Training is full-parameter on 4 GPUs.

```bash
accelerate launch --config_file config/accelerate_qwen35_4gpu.yaml \
  --module tokenar_hoi.token_ar.train --config config/qwen35_4b_behave.yaml
```

Swap the config for OMOMO. Resume with `--resume runs/token_ar/<dataset>/qwen35_4b_correct/checkpoint-XXXXXXX` and the same process count.

The size ablation uses the same tasks, codebooks, and hyperparameters with Qwen3.5-9B-Base (`config/qwen35_9b_behave.yaml`, `config/qwen35_9b_omomo.yaml`). Qwen3.5 has no 8B release. Launch those configs with the same accelerate file. Both pretrained backbones are read from `data/pretrained/Qwen3.5-4B-Base` and `data/pretrained/Qwen3.5-9B-Base`.

## Evaluation

Train the motion–text evaluator on the training split after setting its manifest to the corrected JSONL (`config/evaluator_behave.yaml`, `config/evaluator_omomo.yaml`).

```bash
python -m tokenar_hoi.token_ar.train_evaluator --config config/evaluator_behave.yaml

python -m tokenar_hoi.token_ar.evaluate_codecs \
  --checkpoint runs/token_ar/behave/vq_correct/last.pt \
  --evaluator runs/token_ar/behave/motion_text_evaluator/last.pt \
  --split val --device cuda:0 \
  --output runs/eval_behave_vq

python -m tokenar_hoi.token_ar.benchmark \
  --checkpoint runs/token_ar/behave/qwen35_4b_correct/checkpoint-XXXXXXX \
  --motion-text-evaluator runs/token_ar/behave/motion_text_evaluator/last.pt \
  --split test --temperature 0 \
  --device cuda:0 --dtype bfloat16 \
  --output runs/eval_behave_backbone
```

`benchmark` reports text-to-HOI, object-to-human, and human-to-object. Replace `behave` with `omomo` for the second dataset. `--output` must be a new directory. Metrics are written to `summary.json`.

## Visualization

```bash
python -m tokenar_hoi.token_ar.sample \
  --checkpoint runs/token_ar/behave/qwen35_4b_correct/checkpoint-XXXXXXX \
  --manifest data/processed/behave_correct.jsonl \
  --reference data/parametric_assets/behave/sequences/Date01_Sub01_backpack_back_0.npz \
  --task text_to_hoi --frames 120 \
  --device cuda:0 --dtype bfloat16 \
  --output runs/vis/behave_sample.npz

python -m tokenar_hoi.smpl_mesh \
  --input runs/vis/behave_sample.npz \
  --output runs/vis/behave_sample_mesh.npz
```

`--task` is `text_to_hoi`, `object_to_human`, or `human_to_object`. The mesh archive stores per-frame `vertices` and `faces`.
