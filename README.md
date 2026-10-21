# Unified Controllable HOI — 原生人体参数重训

当前默认训练路径为 `parametric_v1`，按数据集独立训练、采样、评估：

| 数据集 | 人体输出 H | 接触输出 C | 评估 |
| --- | --- | --- | --- |
| BEHAVE | SMPL-H，10 betas + 52 个局部旋转6D + 平移，共325维 | 128维连续 latent → 专属解码器 → 690个表面接触概率 | 原生73关节，TriDi/BEHAVE重建指标 |
| OMOMO | SMPL-X，16 betas + 52 个局部旋转6D + 平移，共331维；面部中性 | 独立128维 latent → 独立解码器 → SMPL-X的690个表面点 | 同语义73关节映射，同一套重建指标公式 |

物体仍为9维刚体姿态；物体几何使用网格BPS。人体解码、接触监督和评估已改为原生网格表面。两个数据集的清单、接触点索引、AE权重、HOI权重和结果目录互相隔离。**现有22关节模型不能直接续训成这个版本，需从头训练。**

**请按 [原生参数重训说明](docs/PARAMETRIC_RETRAINING.md) 操作。** 其中包括当前数据目录、服务器迁移、正式训练、mode010/100评估和完整网格导出。`configs/train_behave.yaml` 与 `configs/train_omomo.yaml` 是新配置，`configs/unified.yaml` 只默认选用OMOMO，不混合两个数据集。

在已配置CUDA环境、迁移好数据和资产的Linux服务器项目根目录运行：

```bash
python -m unified_hoi.train --config configs/train_behave.yaml
python -m unified_hoi.train --config configs/train_omomo.yaml
```

分别启动各自的训练；同一张GPU通常依次执行。正式HOI训练由用户在服务器启动，本机只执行数据准备、接触AE训练及CPU短程验证。当前BEHAVE源手姿是固定值，OMOMO没有手指标注，后者使用显式中性手姿；**支持手指参数不代表已经有动态手指监督**。两套AE独立训练，不使用或声称复现TriDi官方预训练latent坐标。

当前公开入口只运行原生参数模型。旧22关节训练、采样、预处理、控制示例和失效配置已退出运行依赖；按用户要求，回归测试代码和短程验证产物也列入清理清单。实际删除状态见 `docs/code_cleanup_20261021.json`。

## 现在训练需要哪些数据

`data/` 已精简为当前模型依赖，完整目录用途见 [data/README.md](data/README.md)：

- `processed/*_parametric/` 及两份 `.jsonl`：两套原生参数训练数据。
- `processed/object_meshes/`、`processed/text_features/`：清单引用的网格和文本条件。
- `parametric_assets/<dataset>/contact_support.npz` 与 `contact_ae/best.pt`：各自的表面点索引和接触AE。
- `smplx_models/smplh/`、`smplx_models/smplx/`：原生男女身体模型。

将整个 `data/` 保持目录结构迁移到服务器即可。原始BEHAVE、原始文本标注、两套旧body-FK中间数据及旧数据说明已按用户选择删除，不参与当前训练。`OMOMO/` 位于本次清理范围外。

## 代码目录

- `src/unified_hoi/parametric/`：当前模型、扩散、训练、采样、接触AE和评估。
- `src/unified_hoi/`：公开CLI及当前模型复用的注意力层、清单、几何、人体资产和运行工具。
- `scripts/benchmark.py`：批量评估；`prepare_parametric.py`：检查已准备的数据及AE；`cache_text.py`：缓存真实文本条件。
- `configs/`：BEHAVE、OMOMO两份正式配置及默认OMOMO配置。
- `docs/PARAMETRIC_RETRAINING.md`：当前使用说明；`docs/UPSTREAM.md`：保留代码的来源与许可证。

清理脚本默认只预览并校验；执行后移除旧代码、重复压缩包和验证产物。当前数据、两套接触AE、人体资产、环境和原始 `OMOMO/` 均保留。按用户要求，整个 `external/` 目录及其全部参考仓库也保留，脚本禁止删除该目录或其任何子路径。

```powershell
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "D:\code\HOI\scripts\cleanup_current_model.ps1" -Execute
```
