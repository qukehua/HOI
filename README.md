# Unified Controllable HOI — 原生人体参数重训

项目现有两条独立模型路径：原扩散模型 `src/unified_hoi/parametric/` 保持原入口；新建的 `src/unified_hoi/token_ar/` 使用三套VQ-VAE和语言模型进行mask多任务自回归生成。新路线已选定 **Qwen3.5-4B-Base 全参数训练，不使用 LoRA**；两套数据的 `qwen35_4b_*.yaml` 配置、4×32GB启动命令及验证范围见 [两套模型与Token-AR说明](docs/TOKEN_AR.md)。以下说明仍针对原扩散模型。

当前默认训练路径为 `parametric_v1`，按数据集独立训练、采样、评估：

| 数据集 | 人体输出 H | 接触输出 C | 评估 |
| --- | --- | --- | --- |
| BEHAVE | SMPL-H，10 betas + 52 个局部旋转6D + 平移，共325维 | 128维连续 latent → 专属解码器 → 690个表面接触概率 | 原生73关节，TriDi/BEHAVE重建指标 |
| OMOMO | SMPL-X，16 betas + 52 个局部旋转6D + 平移，共331维；面部中性 | 独立128维 latent → 独立解码器 → SMPL-X的690个表面点 | 同语义73关节映射，同一套重建指标公式 |

物体仍为9维刚体姿态；物体几何使用网格BPS。人体解码、接触监督和评估已改为原生网格表面。两个数据集的清单、接触点索引、AE权重、HOI权重和结果目录互相隔离。**现有22关节模型不能直接续训成这个版本，需从头训练。**

**请按 [原生参数重训说明](docs/PARAMETRIC_RETRAINING.md) 操作。** 其中包括当前数据目录、服务器迁移、正式训练、mode010/100评估和完整网格导出。`configs/train_behave.yaml` 与 `configs/train_omomo.yaml` 是新配置，`configs/unified.yaml` 只默认选用OMOMO，不混合两个数据集。

**评估已统一为 `tridi_metrics_v2`，见 [统一评估说明](docs/EVALUATION.md)。** 默认1fps评分、逐帧汇总、MMD/SD平方距离和全局接触F1；支持已有预测复评，不需重新训练。旧指标不能与新口径混用。

在已配置CUDA环境、迁移好数据和资产的Linux服务器项目根目录运行：

```bash
python -m unified_hoi.train --config configs/train_behave.yaml
python -m unified_hoi.train --config configs/train_omomo.yaml
```

分别启动各自的训练；同一张GPU通常依次执行。正式HOI训练由用户在服务器启动，本机只执行数据准备、接触AE训练及CPU短程验证。当前BEHAVE源手姿是固定值，OMOMO没有手指标注，后者使用显式中性手姿；**支持手指参数不代表已经有动态手指监督**。两套AE独立训练，不使用或声称复现TriDi官方预训练latent坐标。

原公开入口运行原生参数扩散模型，Token-AR 使用独立入口。旧22关节实现已清理；本轮继续清理专用测试代码、测试报告和短程验证产物，执行状态统一记录在 `docs/cleanup_tests_docs_20261022.json`。

## 现在训练需要哪些数据

`data/` 已精简为当前模型依赖，完整目录用途见 [data/README.md](data/README.md)：

- `processed/*_parametric/` 及两份 `.jsonl`：两套原生参数训练数据。
- `processed/object_meshes/`、`processed/text_features/`：清单引用的网格和文本条件。
- `parametric_assets/<dataset>/contact_support.npz` 与 `contact_ae/best.pt`：各自的表面点索引和接触AE。
- `smplx_models/smplh/`、`smplx_models/smplx/`：原生男女身体模型。

将整个 `data/` 保持目录结构迁移到服务器即可。原始BEHAVE、原始文本标注、两套旧body-FK中间数据及旧数据说明已按用户选择删除，不参与当前训练。`OMOMO/` 位于本次清理范围外。

## 代码目录

- `src/unified_hoi/parametric/`：原生参数扩散模型、训练、采样、接触AE和评估。
- `src/unified_hoi/token_ar/`：独立的VQ和自回归模型、训练、采样及评估。
- `src/unified_hoi/`：公开CLI及当前模型复用的注意力层、清单、几何、人体资产和运行工具。
- `scripts/benchmark.py`：批量评估；`compare_evaluations.py`：比较同口径评估结果；`prepare_parametric.py`：检查数据及AE；`cache_text.py`：缓存真实文本条件。
- `configs/`：扩散配置；`configs/token_ar/`：VQ、AR和四卡启动配置。
- `docs/PARAMETRIC_RETRAINING.md`、`docs/TOKEN_AR.md`、`docs/EVALUATION.md`：训练、推理和评估说明；`docs/UPSTREAM.md` 与 `docs/TOKEN_AR_ORIGIN.json`：来源和许可证信息。

本轮脚本默认只预览并校验，执行时删除固定清单中的测试代码、报告和验证产物。两条模型路径、运行说明、正式配置、数据、资产和环境均保留。整个 `external/` 及原始 `OMOMO/` 受到保护。历史清理脚本已过期，请使用下面的新命令。

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "D:\code\HOI\scripts\cleanup_tests_docs.ps1" -Execute
```
