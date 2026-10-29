# Unified Controllable HOI — Token-AR

项目只保留 token 自回归模型 `src/unified_hoi/token_ar/`：三套独立 VQ-VAE 把人体 H、物体 O、接触 C 编码为离散 token，语言模型 backbone 在 H/O/C mask 条件下自回归生成，再由 VQ 解码。正式方案为 **Qwen3.5-4B-Base 全参数训练、共享七种 mask，不使用 LoRA**。原连续扩散模型（`parametric/`）已删除，可从 git 历史恢复；其检查点不能用当前代码运行。

模型、mask、四卡配置、训练与推理命令见 [Token-AR 说明](docs/TOKEN_AR.md)。

可选的第三阶段使用小型连续修正网络改善AR粗生成的人体、物体和接触几何。它以真实AR采样缓存训练，冻结Qwen/VQ，并保持已知模态不变；缓存、训练及`--refiner`评估命令见 [连续几何修正](docs/TOKEN_AR.md#可选第三阶段连续几何修正2026-10-29)。尚未完成正式4B检查点的修正训练和效果评估。

| 数据集 | 人体 H | 物体 O | 接触 C |
| --- | --- | --- | --- |
| BEHAVE | SMPL-H，10 betas + 52 个局部旋转6D + 平移，共325维 | 平移3 + 旋转6D | 690个表面接触点 |
| OMOMO | SMPL-X，16 betas + 52 个局部旋转6D + 平移，共331维；面部中性 | 平移3 + 旋转6D | 690个表面接触点 |

两个数据集的 VQ 码本和 AR 检查点互相隔离。物体几何使用网格 BPS，作为8个几何前缀 token。

```bash
# 第一阶段：VQ-VAE
python -m unified_hoi.token_ar.train_codecs --config configs/token_ar/vq_behave.yaml
# 第二阶段：Qwen3.5-4B 四卡全参数
accelerate launch --config_file configs/token_ar/accelerate_qwen35_4gpu.yaml \
  --module unified_hoi.token_ar.train --config configs/token_ar/qwen35_4b_behave.yaml
# 批量评估（tridi_metrics_v2：重建指标 + 1-NNA/COV/MMD/SD）
python -m unified_hoi.token_ar.benchmark --checkpoint runs/behave_10.27/checkpoint-0001000 \
  --split val --output runs/eval_behave_val
```

## 数据

`data/` 已精简为模型依赖，完整说明见 [data/README.md](data/README.md)。Token-AR 读取：

- `processed/*_parametric/` 及对应 `.jsonl` 清单：原生参数数据。
- `processed/object_meshes/`：物体网格（BPS 几何条件）。
- `parametric_assets/<dataset>/contact_support.npz`：690个接触表面点索引。
- `smplx_models/smplh/`、`smplx_models/smplx/`：原生男女身体模型。

## 代码目录

- `src/unified_hoi/token_ar/`：VQ-VAE、AR 模型、训练、采样、单条复评（`evaluate`）和批量基准（`benchmark`）。
- `src/unified_hoi/`：token_ar 复用的清单读取、旋转、网格 BPS、SMPL 网格解码/导出和运行工具。
- `configs/token_ar/`：VQ、Qwen3.5-4B 正式配置、Qwen3-4B/8B 对照配置及四卡 accelerate 配置。
- `docs/TOKEN_AR.md`：训练、推理和评估说明；`docs/TOKEN_AR_ORIGIN.json`：复制来源记录。

整个 `external/` 及原始 `OMOMO/` 受保护，不参与清理。
