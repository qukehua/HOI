# 统一评估协议（2026-10-22，tridi_metrics_v2）

本次只调整评估和新实验的验证配置，不改变模型、训练损失、AE或已有检查点。已有模型不需要重新训练即可复评。

## 固定口径

| 项目 | 默认设置 |
| --- | --- |
| 模式 | H/O/C，1生成、0已知；010=O\|H,C，100=H\|O,C，011=O,C\|H，101=H,C\|O |
| 测试数据 | 当前清单 test split；BEHAVE/OMOMO分别评估 |
| 生成输入 | 原始帧率的120帧窗口；不把输入降到1fps |
| 评分帧 | 1fps；从每条源序列在当前split中第一个可用时间戳起按固定间隔选择，不在片段/窗口边界重新计数 |
| 重建主结果 | 逐帧平均；窗口平均仅作诊断 |
| 接触 | 原生690表面点，完整物体顶点最近距离≤5cm；同时报告全局TP/FP/FN/TN及micro P/R/F1 |
| MMD/SD | 最近邻**平方欧氏距离**均值，与TriDi默认FAISS路径一致 |
| 1-NNA | 生成/参考等量，按样本索引排除自身；理想混合时约0.5 |
| COV | 生成样本命中的唯一参考帧数 / 参考帧总数 |
| NN人体特征 | 73关节减去骨盆，去掉根关节，216维 |
| NN物体特征 | 旋转矩阵9维 + 物体网格几何中心3维；默认减去同一状态的身体骨盆位置 |
| SD训练参考 | train split的10fps真实帧，与TriDi公开配置的训练参考帧率一致 |
| 文本 | 默认关闭；若开启，比较双方和验证均须开启 |
| 采样 | 当前模型DDIM、eta=0，50步，seed=42，每窗口一个样本 |

物体特征使用网格质心，不假定刚体原点就是网格中心。人体生成时不会借用GT人体来设置预测侧的骨盆原点。生成特征的坐标调整只在评分时执行，不改变模型输入或保存的预测。

`--profile tridi` 是有效配置，`auto` 是其别名。`--profile native` 保留全部输入帧，但仍使用修正后的平方距离和按帧主结果。可显式覆盖 `--eval-fps`、`--train-reference-fps`（0=保留全部帧），以及 `--generation-frame native`。帧率必须整除源帧率，不插值或上采样。

## 服务器上重新生成并评估

在项目根目录运行。以下沿用分别选出的best_010/best_100。011沿用事先确定的best_010，101沿用best_100，**不要根据test指标重新挑选检查点**。

```bash
python scripts/benchmark.py \
  --checkpoint runs/behave_parametric_seed42/best_010.pt \
  --manifest data/processed/behave_parametric.jsonl \
  --profile tridi --modes 010 011 --window 120 --steps 50 --seed 42 \
  --device cuda:0 --save-predictions --output runs/eval_behave_v2_object

python scripts/benchmark.py \
  --checkpoint runs/behave_parametric_seed42/best_100.pt \
  --manifest data/processed/behave_parametric.jsonl \
  --profile tridi --modes 100 101 --window 120 --steps 50 --seed 42 \
  --device cuda:0 --save-predictions --output runs/eval_behave_v2_human
```

OMOMO使用其自己的检查点与`data/processed/omomo_parametric.jsonl`，保持其余评估设置相同。输出目录必须是新目录。

## 有保存的预测时无需重新采样

给上述命令加 `--predictions-dir 原评估目录`，使用新的 `--output`。源目录需含`summary.json`及对应的`010/000000.npz`等文件。只填写源目录里实际存在的模式；例如旧目录仅保存010时先用`--modes 010`复评。

复评要求检查点、seed、采样步数、窗口、split、文本条件与原生成过程一致；检查时间戳、模态、资产、AE和已知条件。不能通过复评把“有文本生成”改成“无文本生成”，也不能把101转换成100。新缓存校验检查点SHA256；旧缓存没有SHA256时只能核对文件名及数据/资产，结果明确标记`cached_checkpoint_sha256_verified=false`。

新缓存只保证保存含被选评分帧的窗口。改为更密的评分帧率后，如出现未保存窗口，会报缺失文件，需要重新采样。

## 看哪些文件和字段

- `summary.json.reconstruction[mode]`：主结果；几何误差按帧平均，接触P/R/F1按全局计数计算。
- `summary.json.generation[mode]`：1-NNA/COV/MMD/SD及特征、参考帧数量。
- `summary.json.contact_micro[mode]`：全局接触混淆计数与P/R/F1。
- `summary.json.metrics[mode]`：保留窗口平均/按帧平均的诊断信息，其中窗口F1均值不是全局F1。
- `CONTACT_ALL_NEGATIVE_ACCURACY`：全部预测为无接触的对照准确率。
- `evaluation_frames.jsonl`：每窗口实际使用的源序列、时间戳和帧索引；哈希写入summary。
- `per_clip.jsonl`：各模式每窗口的详细分数。

在两个由本入口生成的报告间做严格检查：

```bash
python scripts/compare_evaluations.py \
  --left runs/run_a/summary.json --right runs/run_b/summary.json --mode 100
```

只比较同一个模式；数据集、帧集合、GT值、协议、文本条件或采样设置不同会拒绝。此工具不会从截图或旧TriDi标量表格推断缺失元信息，也不会把不同采样器的报告标记为完全一致。

## 验证与旧检查点

三份训练配置新增`validation_fps: 1.0`和`validation_text_condition: false`，新实验按此选择best，仍默认固定8个验证窗口。可用相同benchmark命令加`--split val`在完整验证集复评。

旧检查点当时使用的验证口径保留在报告中，不会被伪装成新协议选出的best。续训旧实验请使用该实验保存的原始配置；缺少这两个新字段时保留原有验证行为。不能直接给旧实验替换新配置后继续比较同一套best分数，现有resume检查会拒绝这种配置变更。模型选择只用val，test只用于最终报告。

## 与官方表格的边界

这里对齐了指标定义、默认帧率和汇总方式，**不是声明已经复现官方评估集**。当前数据是经过片段/对象过滤的原生清单，帧选择起点不保证等于官方原始序列起点；骨盆平移归一化不自动重现官方轴向变换、可选朝向/地面对齐和物体模板旋转。OMOMO源fps仍继承原预处理的假设。

所以仍保存`directly_comparable_to_paper_table=false`。正式对比需让TriDi在同一批源帧、同一坐标约定和同一条件下重新评估，并核对其采样预算和检查点选择。旧MMD/SD必须从逐样本特征或预测重新计算，不能对旧均值直接平方。

回归验证：`python -B -m unittest discover -s tests -v`。
