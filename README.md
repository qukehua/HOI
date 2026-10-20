# Unified Controllable HOI

这是基于 **TriDi 的三模态联合扩散思路**与 **Kimodo 的可控动作生成机制**实现的时序 HOI 研究代码。一个模型生成连续的人体、刚体物体与接触序列，并支持在时间、关节和坐标分量上组合条件。

**当前是可训练、可采样、可评估的研究原型，尚没有训练完成的高质量 HOI 权重，也没有动力学控制器或物理可执行性结论。** 历史 CPU 短训练仅检查程序贯通，相关测试产物已清理。本机开发目录为 `D:/code/HOI`；**正式训练在 Linux 服务器上进行**，服务器中的原始 OMOMO 数据放在项目根目录下的 `OMOMO/data`，本项目不修改这些文件。

## 当前实现

- **统一任务**：H/O/C 独立扩散时间步；同一组权重覆盖七种生成/条件组合，以及人体关键帧、根路径、身体部位、物体路径点、旋转和接触区间。
- **精确条件保留**：每步去噪与接触修正后重新写入已知分量。`contact=0` 是明确不接触，未知由 mask 单独表示。
- **时序交互建模**：帧内 H/O/C 注意力、时间注意力；可选根与物体粗阶段，再细化身体；物体局部坐标下的接触关系编码。
- **几何目标与修正**：接触距离、显式非接触间隔、FK 一致性、局部接触滑移和关节地板代理。不能代替网格穿透或物理仿真。
- **真实数据流水线**：OMOMO 原始序列 FK；BEHAVE 30fps 参数与 SMPL-H；时间戳、坐标系、尺度误差、拒绝原因及序列划分均保留。
- **实验基础**：训练集归一化、EMA、断点/RNG 恢复、数据集均衡采样、组合条件留出、消融配置、保存实际采样锚点的评估。

详细说明：[技术路线与研究动机](docs/METHOD.md)、[数据协议](docs/DATA.md)、[控制接口](docs/CONTROLS.md)、[评估口径](docs/EVALUATION.md)、[官方代码接入](docs/UPSTREAM.md)。

人体网格解码使用 OMOMO 的 SMPL-X/16 维 betas 和 BEHAVE 的 SMPL-H/10 维 betas；
资产路径、旧数据元信息补齐、导出命令及位置/FK 一致性方案见 [SMPL 可视化说明](docs/SMPL_VISUALIZATION.md)。

## 训练流程总览

**主线：检查环境 → 选择训练数据 → 小规模检查 → 固定实验配置 → 正式训练 → 续训与保存 → 条件生成 → 测试集评估 → 消融实验。**

这里训练的是新增的统一 H/O/C 时序模型，**从头训练**。不需要先训练 TriDi 或 Kimodo，也不需要把它们的预训练权重加载进来。Kimodo 预训练人体生成属于训练好 HOI 之后可选的输入来源；两篇论文的检查点不兼容本项目网络。

当前在两个数据集上**分开训练、分开验证**，不把 BEHAVE 与 OMOMO 混进同一清单联合训练：

| 路径 | 当前状态 | 训练清单 |
| --- | --- | --- |
| OMOMO | 已转换完成，可直接使用；3797 train / 422 val / 517 test | `data/processed/omomo_with_body_fk.jsonl` |
| BEHAVE | 293条源序列；文本动作片段896 train / 78 val / 480 test | `data/processed/behave_with_body_fk.jsonl`；动作位于 `data/processed/behave_with_body_fk/` |

**以下所有命令均在 Linux 训练服务器的 Bash 终端执行**，从服务器上的项目根目录运行。示例 `/path/to/HOI` 需替换成服务器实际路径，不是 Windows 的 `D:/code/HOI`。已有数据数量和 CPU 检查结果来自本机准备阶段；迁移后须重新检查，**尚未在服务器验证 CUDA 或完成正式训练**。

## 1. 准备 Linux 服务器并检查环境

### 1.1 迁移代码和数据

将当前项目源码、`pyproject.toml`、`configs/`、`scripts/`、`examples/` 等同步到服务器；如果通过 Git 获取，先确保这些文件已提交并推送。测试源码、专用 smoke 配置和历史测试产物列入清理清单；实际执行状态见 `docs/project_cleanup_report.json`。以下目录已被 `.gitignore` 忽略，**不会随 Git clone/pull 下载**：

| 目录 | 服务器上的处理方式 |
| --- | --- |
| `data/processed/` | 复制第2节列出的四个目录和两份清单，保持相对路径 |
| `OMOMO/data/` | 重做 OMOMO 转换时需要；已有完整转换结果时，训练本身不读取原始数据 |
| `data/raw/behave/` | 重建 BEHAVE 时需要原始参数；几何评测需要其中的 `objects/`，已有处理结果的训练无需读取原始参数 |
| `data/smplx_models/` | 复制 SMPL-H / SMPL-X 男女模型及数据检查报告；重复压缩包列入清理清单，见 `docs/project_cleanup_report.json` |
| `data/annotations/` | 重做带文本的转换时复制已有标注映射，或按附录重新准备 |
| `external/` | 可在服务器用 `python scripts/fetch_upstreams.py` 拉取固定版本 |
| `runs/` | 仅需保留既有实验或续训时复制；新实验会自行创建输出目录 |

保持项目内的相对目录结构。现有 OMOMO 清单的 `path` 是相对路径，迁移后不需要把它改成服务器绝对路径；`source_path` 中的 Windows 路径只是来源记录，不用于训练时加载数据。若另行生成的清单使用了绝对 `path` 或 `text_features_path`，应在新实验开始前改为服务器可访问的路径。

**不要复制 Windows 的 `.venv` 作为服务器环境。** 在 Linux 上重新建环境；`requirements-tested.txt` 是本机 CPU 环境的记录，不能作为 Linux CUDA 环境的安装清单。

### 1.2 创建 Python 环境并安装 CUDA PyTorch

服务器需有 Git、Python 和可用的 NVIDIA 驱动。下面以 Python 3.12 为例；若已准备好服务器专用的 Conda/venv 环境，激活它并跳过创建步骤即可，同时把后文的 `source .venv/bin/activate` 换成该环境对应的激活命令。

```bash
cd /path/to/HOI
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

nvidia-smi
```

然后按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)选择 **Linux / Pip / Python** 及与服务器驱动、GPU 兼容的 CUDA 构建，在当前虚拟环境执行页面给出的安装命令。使用当前环境的 `python -m pip`，避免装到系统 Python；确认完成 CUDA PyTorch 安装后，再安装项目：

```bash
python -m pip install -e ".[body]"
python scripts/fetch_upstreams.py
```

新终端都需先 `cd /path/to/HOI` 并 `source .venv/bin/activate`。如使用集群调度系统，GPU 检查和正式训练应在**已分配 GPU 的计算节点/作业内**运行；登录节点检测不到 GPU 不代表训练节点没有 GPU。

### 1.3 在服务器验证实际 GPU 运算

以下检查同时确认操作系统、项目导入路径、CUDA PyTorch 与一次 GPU 前向/反向运算：

```bash
python - <<'PY'
import sys
from pathlib import Path
import torch
import unified_hoi

print("platform:", sys.platform)
print("python:", sys.executable)
print("project:", unified_hoi.__file__)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert sys.platform.startswith("linux"), "Run this check on the Linux training server"
assert Path(unified_hoi.__file__).resolve().parent == Path.cwd().resolve() / "src" / "unified_hoi", "Check the project root and active environment"
assert torch.cuda.is_available(), "CUDA unavailable in this environment or GPU allocation"
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(32, 32, device="cuda:0", requires_grad=True)
(x @ x.T).square().mean().backward()
torch.cuda.synchronize()
print("CUDA forward/backward: OK")
PY
```

最后应输出 `CUDA forward/backward: OK`，才进入正式 GPU 训练；第3步的入口检查不能替代此项。配置里的 `cuda:0` 指当前进程可见的第一个 GPU；调度系统已设置 `CUDA_VISIBLE_DEVICES` 时沿用其分配。独占或手动分配的服务器可在启动前按实际分配设置该变量，本项目仍是单 GPU 训练。

## 2. 只使用当前两份训练数据

本机已完成最终人体 FK 转换，**训练、验证、测试和条件推理统一使用下面两份清单**。无需重新转换、重新缓存文本，或把旧清单合并进来。

| 数据集 | 唯一训练/评估清单 | 动作目录 | train / val / test |
| --- | --- | --- | --- |
| OMOMO | `data/processed/omomo_with_body_fk.jsonl` | `data/processed/omomo_with_body_fk/` | 3797 / 422 / 517 |
| BEHAVE | `data/processed/behave_with_body_fk.jsonl` | `data/processed/behave_with_body_fk/` | 896 / 78 / 480 |

`data/processed` 的运行数据只需要四个目录和两份清单：

```text
data/processed/
├── omomo_with_body_fk.jsonl      OMOMO 入口，内含 train/val/test 划分
├── omomo_with_body_fk/           4736 个最终动作 NPZ
├── behave_with_body_fk.jsonl     BEHAVE 入口，内含 train/val/test 划分
├── behave_with_body_fk/          1454 个最终动作 NPZ
├── text_features/               两数据集共用的 3908 个文本特征
└── object_meshes/               两数据集共用的 31 个物体网格资产
```

旧 `behave/`、`omomo_train/`、`omomo_test/`、`*_with_text.jsonl`、`*_with_mesh.jsonl` 和 `omomo_combined.jsonl` 属于历史中间产物，均不再作为运行入口。清理前逐条核对最终数据已覆盖全部 6190 个旧动作、保留原划分和文本信息，并与记录的旧文件 SHA256 一致。清理执行状态见 `docs/processed_cleanup_report.json`；状态未完成时，旧文件可能仍在磁盘上。

本机 Windows 清理命令如下；不带 `-Execute` 时只校验并预览。脚本按固定清单删除约 435.79 MB 的中间产物，并检查最终数据、原始数据与人体资产未改变。

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "D:\code\HOI\scripts\cleanup_processed.ps1" -Execute
```

最终 NPZ 自带运动、原始性别/体型、模型哈希及原始位置/接触等来源信息。`source_record_path` 只记录转换来源，训练和推理不会沿它打开旧 NPZ。转换报告统一保存在 `docs/data_reports/`，不参与训练数据加载。

### 2.1 迁移与数据检查

向 Linux 服务器复制上面四个目录和两份清单，保持相对目录结构。人体网格导出还需 `data/smplx_models/`；论文指标评估还需 `OMOMO/data` 中的原始测试标注及两数据集的原始物体模型。原始数据和人体资产保留在各自目录，不随本次 processed 清理删除。

```bash
python - <<'PY_CHECK'
from collections import Counter
from pathlib import Path
from unified_hoi.data import read_manifest
for dataset in ("omomo", "behave"):
    records = read_manifest(f"data/processed/{dataset}_with_body_fk.jsonl")
    paths = set()
    for row in records:
        paths.update(row[key] for key in ("path", "object_mesh_path", "text_features_path") if row.get(key))
        paths.update(v["text_features_path"] for v in row.get("text_variants", []) if v.get("text_features_path"))
    missing = [p for p in paths if not Path(p).is_file()]
    print(dataset, dict(Counter(r["split"] for r in records)), "missing:", len(missing))
    assert not missing, missing[:5]
PY_CHECK
```

训练器只读取 train，周期验证只读取 val，benchmark 只读取 test。保留完整三种划分；test 是正式评估数据，不属于测试残留。两个数据集分别训练，不合并清单。

### 2.2 文本条件

两份最终清单已包含文本、全部有效描述和缓存路径。OMOMO 的 4736 条记录中 4669 条有有效文本，共用 103 个 512 维缓存；67 条保持无文本。BEHAVE 的 1454 个片段含 4357 条有效描述，共用 3805 个缓存。缓存总数为 3908，均被最终清单引用。

`configs/train_omomo.yaml` 和 `configs/train_behave.yaml` 中的 `text_condition: true` 启用文本；设为 `false` 可运行无文本条件实验，仍使用同一份最终清单。训练随机选择片段的有效描述，验证和测试固定第一条。单条推理需显式传 `--text-features`；benchmark 需显式传 `--text-conditioned`。

文本标注与时间裁切依据保存在原始标注和 `docs/data_reports/`。需要重建时使用附录中的独立 `data/rebuild/`，不要在日常训练前反复生成旧版 processed 数据。

## 3. 检查训练和推理入口

在服务器已激活的 Python 环境中确认入口能够加载：

```bash
python -B -m unified_hoi.train --help
python -B -m unified_hoi.sample --help
python -B -m unified_hoi.evaluate --help
python -B scripts/benchmark.py --help
python -B scripts/fetch_upstreams.py --verify-only
```

缺少官方源码目录时，先执行 `python scripts/fetch_upstreams.py` 拉取固定版本，再做 `--verify-only`。
入口检查只验证模块导入和参数解析；正式训练与生成仍按后续步骤执行。
2026-10-16 清理清单包括测试代码、专用 smoke 配置和脚本、2–4步测试权重、测试输出及临时文件；实际执行状态见 `docs/project_cleanup_report.json`。如报告状态为 `blocked_by_execution_policy`，文件尚未删除，可在本机 PowerShell 执行 `./scripts/cleanup_project.ps1` 预览，检查后加 `-Execute` 执行。
当前训练使用 `configs/train_omomo.yaml`、`configs/train_behave.yaml` 指向的 `*_with_body_fk.jsonl`；原始数据、完整 train/val/test 划分、文本缓存、网格与人体模型均保留。

## 4. 创建正式实验配置

从默认配置复制一份，给本次实验固定参数；以下名称仅首次使用，已存在则直接检查编辑或换新名字：

```bash
cp -n configs/unified.yaml configs/train_omomo.yaml
cp -n configs/unified.yaml configs/train_behave.yaml
```

分别编辑对应配置。仓库已提供 `configs/train_omomo.yaml` 与 `configs/train_behave.yaml`，两者均使用 `text_condition` 开关，并已接入文本缓存。做无文本基线时将同一配置的开关改为 `false`：

| 参数 | 起始设置 | 含义与注意点 |
| --- | --- | --- |
| `manifest` | OMOMO：`data/processed/omomo_with_body_fk.jsonl`；BEHAVE：`data/processed/behave_with_body_fk.jsonl` | 两个数据集分开训练，各自用自己的清单；两种文本设置共用同一数据集的带缓存清单 |
| `text_condition` | 两者均为 `true` | 是否在训练与验证中输入文本；关闭则运行相同片段上的无文本基线 |
| `device` | `cuda:0` | 正式实验明确指定GPU；默认 `auto` 在没有CUDA时会退回CPU |
| `seed` | `42` | 不同随机种子作为独立实验 |
| `window / stride` | `120 / 60` | 10fps下最多12秒窗口、6秒步进；短片段用有效帧mask补齐 |
| `batch_size` | `8` | 起始值，未验证显存要求；显存不足时在新实验中降至4或2 |
| `workers` | `0` | 先使用单进程读取排查数据；Linux 上可再按 CPU、内存与存储吞吐调整并行读取 |
| `learning_rate` | `0.0001` | AdamW固定学习率，当前没有学习率调度器 |
| `max_steps` | `100000` | 总优化步数，是初始预算，不是保证收敛的步数 |
| `balance_datasets` | `true` | 单数据集清单时等价于均匀采样；若清单内出现多来源标签再按窗口数反比加权 |
| `diffusion_steps` | `1000` | 训练噪声时间步数；与推理 `--steps 50` 不同 |
| `geometry_weight` | `0.05` | 几何损失权重 |
| `ema_decay` | `0.999` | 采样/评估默认使用EMA权重 |
| `save_every / validate_every` | `1000 / 1000` | 每1000步保存并验证，达到max_steps时也保存 |
| `validation_batches` | `20` | 周期验证仅前20个val batch，不是完整验证集评估 |
| `model.width / heads / layers` | `192 / 6 / 4` | 统一模型规模 |
| `model.relations / two_stage` | `true / true` | 关系分支及根/物体粗阶段 |
| `holdout_signatures` | `[]` | 常规训练不留出；组合泛化实验另设 |

当前训练器使用**单设备、FP32**，尚无DDP、AMP或梯度累积。不要把扩大GPU数量当作现有脚本自动支持的功能。未提供可靠训练耗时或显存估计；先做下一步的小预算检查。

## 5. 启动正式训练，并检查日志

普通 SSH 服务器上建议在 `tmux` 会话中运行长任务，避免断开终端导致训练退出；若服务器使用 Slurm 等调度系统，应通过其 GPU 作业执行下面的训练命令，`tmux` 不能替代资源分配。

```bash
# 可选：服务器已安装 tmux 时使用
tmux new -s hoi_train
# 进入新会话后，重新定位目录并激活环境
cd /path/to/HOI
source .venv/bin/activate
```

运行期间按 `Ctrl+B`，松开后按 `D` 可分离会话；重连 SSH 后用 `tmux attach -t hoi_train` 返回。每个命令块中的 `\` 是 Bash 续行符，必须是该行最后一个字符，后面不要加空格。

先在完整数据、正式模型和GPU上跑1000步，验证显存、日志、周期验证与保存均正常。

OMOMO：

```bash
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"

python -m unified_hoi.train \
  --config configs/train_omomo.yaml \
  --output runs/omomo_seed42 --max-steps 1000
```

BEHAVE（与 OMOMO 分开训练，使用独立配置与输出目录）：

```bash
python -m unified_hoi.train \
  --config configs/train_behave.yaml \
  --output runs/behave_text_seed42 --max-steps 1000
```

OMOMO 和 BEHAVE 直接使用各自配置里的带缓存清单，通过 `text_condition` 开关选择文本输入。比较两种设置时分别指定输出目录；续训必须保持清单和开关不变。

首次训练会先扫描train窗口并计算归一化统计，之后才开始打印优化损失；这阶段可能没有逐步日志。归一化不会使用val/test。训练中的随机mask自动覆盖多种条件，不需要给human→object、object→human分别启动不同模型。

运行目录应包含：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 实际生效的配置，含命令行覆盖值 |
| `run_info.json` | 设备、参数量、训练窗口数、来源版本 |
| `train.jsonl` | 总损失、H/O/C去噪损失、几何项、梯度范数、实际观察签名 |
| `validation.jsonl` | EMA模型的周期验证损失及验证batch数量 |
| `last.pt` | 模型、EMA、优化器、归一化、步数、数据进度和随机状态 |

检查 `run_info.json` 确实使用CUDA。日志默认每20步记录一次，也记录第1步；损失应有限，观察若干验证周期的趋势，不能只凭总损失下降或锚点误差为0判断质量。

```bash
cat runs/omomo_seed42/run_info.json
tail -n 5 runs/omomo_seed42/train.jsonl
tail -n 5 runs/omomo_seed42/validation.jsonl

# BEHAVE 同理，把目录换成 runs/behave_seed42
cat runs/behave_seed42/run_info.json
```

1000步完成后，先保留这个检查点，再继续到总计100000步。以 OMOMO 为例（BEHAVE 把配置、清单与输出目录换成 `train_behave.yaml` / `behave_with_body_fk.jsonl` / `runs/behave_text_seed42`）：

```bash
cp -n runs/omomo_seed42/last.pt runs/omomo_seed42/step_001000.pt

python -m unified_hoi.train \
  --config configs/train_omomo.yaml \
  --output runs/omomo_seed42 \
  --max-steps 100000 \
  --resume runs/omomo_seed42/last.pt
```

当前只自动更新 `last.pt`，没有自动生成 `best.pt` 或保留全部历史检查点。需要比较训练阶段时，应在训练暂停且保存完成后另存检查点；模型选择依据val，最终报告再使用test，避免用test反复选模型。

## 6. 断点续训规则

中断后重复上面的 `--resume` 命令。它恢复优化器、EMA、随机状态和epoch内进度；不是仅加载模型权重。

- `--max-steps` 指**累计总步数**，例如从1000续到100000，而不是额外训练100000步。
- 必须保持模型结构、数据清单路径与内容、窗口、batch size、种子、学习率、损失、mask分布等不变。程序会拒绝关键配置不一致；不要用更改文件内容绕过检查。
- 若第一次用了 `--manifest` 或 `--output`，续训也显式保留这些参数，防止读回默认路径。
- 更换数据集、加入文本缓存、切换 `text_condition` 或更改网络结构属于新实验；当前没有单独的“仅加载权重后微调”CLI。
- 在第一次保存之前中断，没有可恢复检查点。普通中断不会自动额外保存，最多丢失上次保存后的更新。
- 已有 `last.pt` 的输出目录不能不带 `--resume` 再启动，以免覆盖已有训练。

## 7. 用训练后的检查点生成交互

以下使用真实存在的test序列验证推理路径；同样的采样命令也可改用val序列做开发期观察。**检查点路径与数据清单要换成自己的实验，不要使用4步smoke权重做效果结论。**

人体轨迹已知，生成物体与接触：

```bash
python -m unified_hoi.sample \
  --checkpoint runs/omomo_seed42/last.pt \
  --reference data/processed/omomo_with_body_fk/004219_omomo_sub16_clothesstand_000.npz \
  --object-mesh data/processed/object_meshes/omomo_clothesstand_43d7a5221325bb58_v1.npz \
  --object-mesh-scale 0.27460962533950806 \
  --mode 011 --frames 120 --steps 50 --seed 42 \
  --output runs/omomo_seed42/human_to_object.npz
```

`mode` 的三位按H/O/C排列，`1=生成，0=给定`。例如 `011` 为H→O+C，`101` 为O→H+C，`111` 为无H/O/C观测的联合生成；后者仍以物体几何和骨架为条件。短于120帧的参考片段会使用其实际长度。

混合人体关键帧、物体路径点与接触区间：

```bash
python -m unified_hoi.sample \
  --checkpoint runs/omomo_seed42/last.pt \
  --reference data/processed/omomo_with_body_fk/004219_omomo_sub16_clothesstand_000.npz \
  --object-mesh data/processed/object_meshes/omomo_clothesstand_43d7a5221325bb58_v1.npz \
  --object-mesh-scale 0.27460962533950806 \
  --controls examples/mixed_controls.json \
  --frames 120 --steps 50 --projection-steps 30 --seed 42 \
  --output runs/omomo_seed42/mixed.npz
```

示例JSON中的帧号主要在0–7，只演示接口；正式实验应按自己的目标编辑，不能把它当作该序列的真实接触标注。`--projection-steps 30` 是采样后的几何修正，不是物理仿真。输出NPZ同时保存实际观测值/mask，旁边的 `.metrics.json` 保存诊断；文件名需使用小写 `.npz` 且不能已存在。

文本训练的模型如需文本条件，采样还要传 `--text-features path/to/embedding.npy`（同一CLIP编码器的512维特征）；单独指定reference不会自动读取文本缓存。benchmark还需显式传 `--text-conditioned` 才会使用清单中的文本缓存。

不需要目标动作也可以生成：`--scene scene.npz --frames 120 --controls ...`，scene包含物体点云、骨架偏移与fps。`examples/scene_controls.json` 使用明确坐标；`from_reference: true` 仅在确实有参考动作时使用。控制定义详见 [CONTROLS.md](docs/CONTROLS.md)。

## 8. 在测试集上做定量评估

默认按数据集选择指标：BEHAVE 使用 **TriDi**（模式 `011`：OBJ_V2V、OBJ_CENTER、CONTACT_*，并计算 generation 指标）。OMOMO 仍使用 **Uni-HOI（arXiv:2604.27491v2）**。BEHAVE 改回 Uni-HOI 时传 `--profile uni-hoi`。

| 任务 | 数据集 | 模式 | 默认主指标 |
| --- | --- | --- | --- |
| 物体动作 → 人体动作 | OMOMO / FullBodyManipulation | `101` | Uni-HOI：HandJPE、MPJPE（cm）、C_prec、C_rec、C_acc、c% |
| 人体动作 → 物体动作 | BEHAVE | `011` | TriDi：OBJ_V2V、OBJ_CENTER、CONTACT_* |
| 文本 → HOI | BEHAVE、OMOMO | `111` + 文本，且 `--profile uni-hoi` | FID、R-Precision Top-1/2/3、Diversity |

其中 HandJPE 使用世界坐标；MPJPE 使用各自骨盆对齐后的24关节。接触指标根据双手与原始物体网格顶点的5cm距离重新计算，不使用模型的22维接触输出，也不只检查已知锚点。c%按论文表格输出0–1比例。E_v2v使用对应的完整mesh顶点；E_ch使用每侧10000个独立表面采样点，计算双向非平方距离之和。论文正文写E_ch、表3写E_c，代码保留明确的E_ch名称，另附三维质心误差，避免混淆。

几何指标已接通；FID等特征指标的计算代码已实现，但**Uni-HOI配套的预训练HOI/文本评估器尚未取得和验证**。不能用原始关节、模型隐藏层或仅CLIP文本特征替代。当前数据划分、10fps窗口、采样次数等也尚未与论文完整核齐，因此指标名称和公式对齐不等于论文表格数值可直接对比。详细定义、依赖和差异见 [EVALUATION.md](docs/EVALUATION.md)。

Linux服务器评估时还需复制原始几何/评估资产（已转换的训练NPZ不用重做）：

```text
data/raw/behave/objects/
OMOMO/data/captured_objects/
OMOMO/data/test_diffusion_manip_seq_joints24.p
```

这些路径可用 `--behave-objects`、`--omomo-objects`、`--omomo-raw` 覆盖。原始OMOMO文件用于取得真实24关节骨架偏移和逐帧物体尺度；只在评价阶段读取，不传给模型作新条件。

先用少量窗口检查，再去掉 `--limit` 跑完整test，并换新输出目录：

```bash
python scripts/benchmark.py \
  --checkpoint runs/omomo_seed42/last.pt \
  --manifest data/processed/omomo_with_body_fk.jsonl \
  --modes 101 --window 120 --steps 50 --seed 42 --limit 2 \
  --output runs/eval_omomo_seed42_check
```

下面分别评估两个数据集。检查点与清单必须来自**同一数据集上的训练**，不要用 OMOMO 检查点去评 BEHAVE，或反过来：

```bash
python scripts/benchmark.py \
  --checkpoint runs/omomo_seed42/last.pt \
  --manifest data/processed/omomo_with_body_fk.jsonl \
  --modes 101 --window 120 --steps 50 --seed 42 \
  --output runs/eval_omomo_seed42_raw --save-predictions

python scripts/benchmark.py \
  --checkpoint runs/behave_seed42/last.pt \
  --manifest data/processed/behave_with_body_fk.jsonl \
  --window 120 --steps 50 --seed 42 \
  --output runs/eval_behave_seed42_raw --save-predictions
```

不指定 `--profile` 时，BEHAVE 用 TriDi，其它数据集用 Uni-HOI。不指定 `--modes` 时，会按记录的 dataset 字段自动对 OMOMO 跑 101、对 BEHAVE 跑 011，并按数据集/任务分组汇总。默认关闭文本条件，即使清单存在文本缓存也不会自动使用。对OMOMO复现“物体+文本→人体”设置时，需要文本训练的检查点、带缓存清单及 `--text-conditioned`。BEHAVE 的文本生成任务不是 TriDi 单目标模式，需要 `--profile uni-hoi --modes 111 --text-conditioned`。

输出 `per_clip.jsonl` 和 `summary.json`。主指标位于summary的 `metrics`；按窗口宏平均和按帧加权均值分列，原有锚点、FK、滑移、地板及耗时诊断位于 `diagnostics_by_dataset_mode`。不要混用两种聚合方式。OMOMO原版代码支持best-of-20；本项目默认单样本，需要这一对照时传 `--samples-per-window 20 --modes 101`，以最小MPJPE选出同一个样本计算所有指标，并如实报告采样预算。Uni-HOI是否使用相同预算尚未确认。

对比几何修正时，保持检查点、数据、窗口、种子和采样步数一致，另跑 `--projection-steps 30` 并使用新目录。硬锚点误差为0只证明条件被保留；它不等于生成质量、接触正确或物理可执行。

文本生成HOI需要单独进行数据集级评估：

```bash
python scripts/benchmark.py \
  --checkpoint runs/omomo_text_seed42/last.pt \
  --manifest data/processed/omomo_with_body_fk.jsonl \
  --modes 111 --text-conditioned --window 120 --steps 50 --seed 42 \
  --output runs/eval_text_hoi
```

这会保存生成结果和 `feature_inputs.jsonl`，单条结果中的FID等保持null并注明缺少评估器。用与对照方法一致的已训练HOI/文本评估器提取配对特征后，可运行：

```bash
python -m unified_hoi.evaluate_features \
  --features outputs/hoi_evaluator_features.npz \
  --benchmark runs/eval_text_hoi \
  --retrieval-batch-size 32 --diversity-pairs 300 --seed 42 \
  --output runs/eval_text_hoi/feature_metrics.json
```

32/300仅为命令示例，论文没有明确这两个设置；应换成对照评估器的实际协议。NPZ特征格式及评估器来源记录要求见 [EVALUATION.md](docs/EVALUATION.md)。Diversity应接近真实数据的值，不是越大越好。

我们自己的稀疏混合控制实验继续使用辅助诊断模式：

```bash
python scripts/benchmark.py \
  --profile diagnostics --modes mixed human_object \
  --checkpoint runs/omomo_seed42/last.pt \
  --manifest data/processed/omomo_with_body_fk.jsonl \
  --window 120 --steps 50 --seed 42 \
  --output runs/eval_control_seed42
```

单条样例仍可运行 `python -m unified_hoi.evaluate --prediction ... --reference ... --output ...`。
默认根据保存的完整条件mask识别论文任务，并额外输出 `paper_evaluation`；混合控制保留原有诊断。要只检查控制条件，传 `--profile diagnostics`。不把参考序列自动当成已知条件。


## 9. 做消融与未见条件组合实验

先完成一组正常训练，再启动研究比较。用已确定的正式配置生成消融配置：

```bash
python scripts/make_ablation_configs.py \
  --base configs/train_omomo.yaml --output configs/ablations_omomo
```

脚本生成六个文件，每个都要单独训练；不是运行脚本就完成了消融：

| 配置 | 关系分支 | 粗阶段 | 留出实际H+O观察组合 |
| --- | --- | --- | --- |
| `mask.yaml` | 关 | 关 | 否 |
| `mask_coarse.yaml` | 关 | 开 | 否 |
| `relation.yaml` | 开 | 开 | 否 |
| 上述各自的 `*_heldout_HO.yaml` | 同上 | 同上 | 是 |

例如：

```bash
python -m unified_hoi.train \
  --config configs/ablations_omomo/mask.yaml \
  --manifest data/processed/omomo_with_body_fk.jsonl \
  --output runs/ablations_omomo/mask_seed42

python -m unified_hoi.train \
  --config configs/ablations_omomo/relation_heldout_HO.yaml \
  --manifest data/processed/omomo_with_body_fk.jsonl \
  --output runs/ablations_omomo/relation_heldout_HO_seed42
```

对留出模型评估稀疏H+O条件：

```bash
python scripts/benchmark.py \
  --checkpoint runs/ablations_omomo/relation_heldout_HO_seed42/last.pt \
  --profile diagnostics \
  --manifest data/processed/omomo_with_body_fk.jsonl \
  --window 120 --steps 50 --seed 42 \
  --modes human_object --observed-signatures H+O \
  --projection-steps 0 --output runs/eval_heldout_HO_seed42
```

`human_object` 是稀疏H+O约束；`001` 的完整H/O都给定，主要在推断接触，不能替代上述生成实验。`human_contact`、`object_contact` 是另外两种稀疏组合，`mixed` 是H+O+C。

各模型都比较不修正/相同几何修正，保持数据、训练步数、随机种子集合和模型宽度一致，并报告参数量与耗时。多个训练种子必须在各自YAML中更改 `seed`、使用新目录重新训练；benchmark的 `--seed` 只控制评估条件/采样，不能代替多个训练种子。若训练命令使用了 `--manifest` 覆盖，后续生成消融配置和训练时也要保持同一数据版本。

## 可选：接入 Kimodo 生成人体

训练新HOI模型不依赖这一阶段。希望“先由Kimodo生成人体，再补全O/C”时，按 [UPSTREAM.md](docs/UPSTREAM.md) 安装官方推理依赖及对应模型资产，先运行官方推理桥，然后转换：

```bash
python -m unified_hoi.integrations.kimodo convert \
  --input outputs/kimodo/human.npz \
  --output outputs/kimodo/human_reference.npz --target-fps 10
```

这条命令需要已有真实Kimodo输出；它本身不会生成人体。桥接执行整数stride下采样，保留源时间戳，并反解/检查静态骨架。随后在HOI采样时传 `--human-prior outputs/kimodo/human_reference.npz`，使身体尺寸与Kimodo人物匹配。要求帧数、fps及用户期望的世界坐标一致。当前没有运行真实预训练Kimodo推理。

## 附录：需要时在独立目录重建

已有第2节的数据时跳过本节。以下命令只用于从原始数据重建新版本；中间文件全部写入 `data/rebuild/`，已有输出不会覆盖。完成后核验新版本，再迁移最终四个目录和两份清单；训练中的实验继续使用原数据版本。

OMOMO：

```bash
python scripts/fetch_upstreams.py
python scripts/prepare_omomo_text.py
python -m unified_hoi.preprocess omomo --input OMOMO/data/train_diffusion_manip_seq_joints24.p --objects OMOMO/data/captured_objects --source-fps 30 --target-fps 10 --points 1024 --text-file data/annotations/omomo_text.json --val-from-train 0.1 --split-seed 0 --fps-provenance "Assumed 30 fps; not verified by release metadata" --output data/rebuild/omomo_train
python -m unified_hoi.preprocess omomo --input OMOMO/data/test_diffusion_manip_seq_joints24.p --objects OMOMO/data/captured_objects --source-fps 30 --target-fps 10 --points 1024 --text-file data/annotations/omomo_text.json --fps-provenance "Assumed 30 fps; not verified by release metadata" --output data/rebuild/omomo_test
python scripts/combine_manifests.py data/rebuild/omomo_train/manifest.jsonl data/rebuild/omomo_test/manifest.jsonl --output data/rebuild/omomo_combined.jsonl
python scripts/cache_text.py --manifest data/rebuild/omomo_combined.jsonl --output data/rebuild/omomo_with_text.jsonl --device cpu
python scripts/prepare_object_meshes.py --manifest data/rebuild/omomo_with_text.jsonl --output data/rebuild/omomo_with_mesh.jsonl
python scripts/prepare_body_metadata.py --manifests data/rebuild/omomo_with_mesh.jsonl --write
python scripts/prepare_body_geometry.py --manifest data/rebuild/omomo_with_mesh.jsonl --output data/rebuild/omomo_with_body_fk.jsonl
```

BEHAVE：

```bash
python -m unified_hoi.preprocess behave --input data/raw/behave --objects data/raw/behave/objects --smpl-models data/smplx_models --split-file data/raw/behave/split.json --val-from-train 0.1 --split-seed 0 --target-fps 10 --points 1024 --output data/rebuild/behave_source
python scripts/prepare_behave_text.py --manifest data/rebuild/behave_source/manifest.jsonl --output data/rebuild/behave
python scripts/cache_text.py --manifest data/rebuild/behave/manifest.jsonl --output data/rebuild/behave_with_text.jsonl --device cpu
python scripts/prepare_object_meshes.py --manifest data/rebuild/behave_with_text.jsonl --output data/rebuild/behave_with_mesh.jsonl
python scripts/prepare_body_metadata.py --manifests data/rebuild/behave_with_mesh.jsonl --write
python scripts/prepare_body_geometry.py --manifest data/rebuild/behave_with_mesh.jsonl --output data/rebuild/behave_with_body_fk.jsonl
```

原始 OMOMO 位于 `OMOMO/data`，BEHAVE 位于 `data/raw/behave`，人体模型位于 `data/smplx_models`，文本标注位于 `data/annotations`。这些目录保留。初次下载、时间戳、尺度、坐标和划分协议见 [DATA.md](docs/DATA.md)。

## 常见问题与当前边界

| 现象 | 处理方式 |
| --- | --- |
| CUDA不可用/装的是CPU PyTorch | 在已获 GPU 分配的 Linux 节点检查驱动、激活环境和 PyTorch CUDA 构建；按第1步实际运行 GPU 运算 |
| 迁移后提示文件不存在 | 从服务器项目根目录执行；确认已单独复制被 Git 忽略的数据、文本缓存及模型，并保留清单对应的目录结构 |
| 显存不足 | 开新实验前降低batch size；再考虑窗口长度，并让比较实验保持相同设置 |
| `Training checkpoint already exists` | 保持配置使用 `--resume`，或换新输出目录 |
| `Resume configuration changed ...` | 对照该run的 `config.json` 恢复原参数；换数据/结构应另开实验 |
| `No training windows` 或转换0条 | 检查manifest、split与转换报告；不要把test重新标成train来绕过错误 |
| BEHAVE找不到SMPL-H | 补齐持证资产并修正 `--smpl-models`；已下载动作参数不能替代身体模型 |
| 已有文本但效果像无文本 | 检查使用的manifest是否含有效 `text_features_path`；单条采样还需明确传文本特征 |
| 日志开头暂时没有loss | 首次扫描和训练集归一化在优化步骤前执行 |
| 锚点误差0但人体/物体仍不合理 | 继续检查接触几何、FK、未知部分和模型训练状态；硬条件保持不代表交互成功 |

训练好当前模型后得到的是**运动学参考轨迹**。物理可执行HOI还需要带质量/惯量和关节限制的仿真模型、闭环跟踪器及真实执行反馈；接触距离、地板代理和滑移指标不能代替它们。研究动机、限制和本次已验证范围见 [METHOD.md](docs/METHOD.md) 与 [IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md)。

## Mesh 物体几何（2026-10-14）

当前正式配置使用 `model.geometry_encoder: mesh_bps`：物体模板 mesh 的固定顺序
1024×3 BPS 位移经 3072→768→width MLP 编码。参考 InfBaGel 的 BPS 与顶点接触做法，
这是独立实现；没有加载其模型权重或复用其场景/consistency model。

当前最终清单已经关联 `data/processed/object_meshes/` 并完成 FK 转换；日常训练直接使用第2节的数据。需要重建时按附录在 `data/rebuild/` 生成中间文本/mesh 清单。
新网络需新建实验从头训练，不能续训旧 pointnet 权重。旧模型配置未声明 geometry_encoder 时仍加载 pointnet，便于旧结果复查。

接触支持使用最多10000个确定性 mesh 顶点；少于此数重复填充，不改变最近距离。
加载时以同一几何重新生成22关节接触代理，保持原数据阈值；不会把22关节伪称为手掌。
当前配置启用 contact、noncontact、slip、fk；几何项基于驱动网格的 FK 位置。mesh 正接触损失有2cm容差，显式非接触仍用
记录中的接触阈值。它们是运动学代理，不是三角面精确距离、SDF或网格穿透保证。
采样时 `--projection-steps 30` 可开启已有的最后几何修正；没有移植上游逐步引导。

benchmark 使用新清单即可自动加载 mesh。单条采样使用清单对应的 `object_mesh_path`
和 `object_mesh_scale`：

```bash
python -m unified_hoi.sample --checkpoint runs/omomo_body_fk_seed42/last.pt --reference <motion.npz> --object-mesh <canonical_mesh.npz> --object-mesh-scale <recorded_scale> --mode 101 --projection-steps 30 --output <new_output.npz>
```

BEHAVE 的规范 mesh 已减去原模板顶点均值、scale=1；OMOMO 保留标注原点并使用序列记录
的中位尺度。mesh 在物体局部坐标中，不重复做世界坐标旋转。清单构建会与旧转换的表面
采样逐项比对，原点或尺度不匹配时拒绝。新采样文件保存实际接触顶点和BPS，可复查诊断。

参考：<https://github.com/YudeZou/InfBaGel-release/blob/main/code/models/infbagel.py>，
<https://github.com/YudeZou/InfBaGel-release/blob/main/code/guidance_loss.py>。

## 人体 FK 与 SMPL 一致性（2026-10-16）

本机已完成 4736 条 OMOMO 和 1454 条 BEHAVE 的人体 FK 转换，正式配置直接使用两份 `*_with_body_fk.jsonl`。转换使用实际人体资产和原始体型重建骨长、xyz、接触，最终 NPZ 内保存原始位置、骨架、接触和来源文件哈希。旧中间 NPZ 已列入清理清单；需要重建时按附录操作。

新训练会重算训练集归一化；请使用正式配置的新输出目录，不能 `--resume` 旧 checkpoint。
资产 SHA256 和数据版本随预测保存，渲染时核对。旧模型仍可采样，但后处理不能代替新训练。
最终输出始终为 FK 位置；非根位置锚点默认由 `--ik-steps 90` 求解，超过 1 mm 残差明确报告，
不会再用覆盖 xyz 伪造满足。`--projection-steps` 控制额外接触优化。
完整迁移方式和验证范围见 [SMPL 可视化说明](docs/SMPL_VISUALIZATION.md)。
