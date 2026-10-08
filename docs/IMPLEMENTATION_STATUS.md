# 本次交付与验证状态

交付目录：`D:/code/HOI`。用户原有 `OMOMO/` 保留。本机 `.venv/` 和 `requirements-tested.txt` 仅用于 CPU 程序检查；正式训练在 Linux 服务器另建 CUDA 环境，见 README。本次没有获得或训练好正式 HOI 模型。

## 数据实际状态

| 数据 | 已完成 | 尚需处理 |
|---|---|---|
| BEHAVE | 299条源序列已处理；293条完成真实 SMPL-H 转换，共147520帧、18种交互物体；6条因不在官方划分中而跳过；逐条数据校验通过 | 迁移到 Linux 后单独训练与验证 |
| OMOMO | 已读取用户解压数据；全量转换并保留4736条序列、291801帧 | 源fps仍为30的显式假设，未由发布元数据确认；与 BEHAVE 分开训练 |
| OMOMO文本 | 保留数据中4669条有匹配文本，103个512维CLIP缓存已生成；原配置通过 `text_condition` 开关启用 | 其余67条无文本；未正式训练文本模型 |
| BEHAVE文本 | HOI-Diff全部1613个文件、4835条描述已下载并校验；1454个动作片段、4357条有效描述已对齐，3805个512维CLIP缓存已生成 | 293条源序列划分保持不变；排除和时间裁切原因见 `behave/preparation_report.json` |

完整 OMOMO 划分为 **3797 train / 422 val / 517 test**；先按源序列分组，再生成时间窗口。测试受试者sub16/17没有进入train/val。原始清单位于 `data/processed/omomo_combined.jsonl`，默认训练配置使用带缓存的 `data/processed/omomo_with_text.jsonl`。

原始训练来源5280条中保留4219条，拒绝1061条；原始测试602条中保留517条，拒绝85条。拒绝范围包含mop/vacuum多部件物体及超出默认1%尺度波动阈值的序列。固定中位数尺度后的完整物体表面几何改变上界，保留序列中最大约 **8.442 mm**。详情和逐条原因见两个 `conversion_report.json`，这不是无损恢复全部原始标注。

SMPL-H 不随本代码分发，下载资产仍被 Git 忽略。男女 `.pkl` 模型保留在 `data/smplx_models/smplh/`，重复的 `smplx.zip` 已移出 `data` 归档，路径见 `docs/data_cleanup_report.json`。真实模型以 `num_betas=10, use_pca=False` 通过 CPU 加载及前向检查，网格形状均为 `[1,6890,3]`；下载检查见 `data/smplx_models/smplh_download_report.json`。

历史全量转换在本机 CPU 完成：10fps、1024点、194 train / 17 val / 82 test，共293条源序列。当前 `data/processed/behave/` 保存的是从这些源序列裁出的1454个文本片段，原293条完整转换文件已不在该目录。原始参数、物体网格、模型和文本标注均保留，需完整源序列时可按README重建到 `data/processed/behave_source/`。当前训练入口为 `data/processed/behave_with_text.jsonl`，与 OMOMO 分开训练/验证。

## 已跑过的流程

BEHAVE文本片段清单为 `data/processed/behave_with_text.jsonl`，896 train / 78 val / 480 test，分别继承194 / 17 / 82条源序列。仍使用 `configs/train_behave.yaml`，通过 `text_condition` 开关控制文本输入；训练随机选择同一片段的有效描述，验证和测试固定第一条。下载、对齐和缓存重建命令见README第2.3节。

全部1454个动作文件与源时间戳切片逐项核对通过，3805个缓存均为有效非零512维向量。真实小样本完成2步CPU训练、2次验证和测试集文本条件采样；文本分支梯度非零，清零文本会改变损失。含多描述的训练跨epoch及epoch内断点续训均与连续训练逐项一致。报告：`data/processed/behave/verification_report.json`，测试采样：`runs/verification/behave_text_test_smoke/summary.json`；均非正式质量实验。

- 真实 OMOMO 小样本转换：原train来源20条划为18 train / 2 val，另取官方test10条，三者没有混用。
- CPU小模型（57,753参数）训练4步：损失/梯度有限，保存模型、EMA、优化器、归一化与RNG状态；不是正式收敛实验。
- 一个检查点覆盖七种完整生成方向，以及稀疏混合H/O/C约束；检查锚点逐分量精确保留、输出有限。
- CLI生成、保存实际锚点、离线评估；无目标动作的scene-only输入也完成采样。
- 评估程序已在测试窗口上运行，并单独检查稀疏H+O模式的实际条件签名。小样本统计不构成泛化结论。
- 断点恢复测试比较连续训练与2→3步恢复，包括跨epoch情况；模型、EMA、优化器和RNG一致。
- 官方源码commit、remote、工作区洁净状态已核对；TriDi复制函数与原始源码AST相同。
- Kimodo接口以官方签名与导出字段核对，并用测试替代模型验证桥接；**没有下载或运行预训练Kimodo权重**。
- Kimodo桥接支持真实整数stride降采样、源时间戳、静态骨架反解和全源帧FK检查；HOI接入使用该人物自己的骨架尺寸。

原交付单元/集成测试132项通过，历史结果保存在 `runs/verification/tests.xml`。2026-09-30加入Uni-HOI评价指标后，全套测试 **155项通过**。原程序报告：`runs/final_smoke/samples/report.json`、`runs/smoke_verified/benchmark/summary.json`、`runs/smoke_verified/benchmark_heldout_pairs/summary.json`。

## Uni-HOI评价指标更新（2026-09-30）

- 默认benchmark按论文任务报告主指标：OMOMO物体→人体的HandJPE/MPJPE（cm）、接触精确率/召回率/准确率/接触帧比例；BEHAVE人体→物体的E_ch/E_v2v（m）。混合条件诊断保留。
- OMOMO评价恢复原始24关节骨架，使用真实物体网格顶点与逐帧尺度；HandJPE为世界坐标，MPJPE减去各自骨盆。已与官方OMOMO函数对照验证数值。
- 原始训练NPZ和模型结构没有改变。评价需额外读取已有原始OMOMO测试joblib及两数据集的物体mesh。
- FID、R-Precision Top-1/2/3、Diversity的特征计算器、配对输出及来源检查已实现；没有取得/验证Uni-HOI配套的预训练HOI-文本评估器，尚不能产出可信的论文可比文本生成分数。缺失时明确输出null与原因。
- 论文正文E_ch和表3的E_c命名不一致：使用正文明确的双向Chamfer定义，质心误差单独报告，不混为同一个指标。
- 两数据集各一条真实测试窗口完成新benchmark生成→评价→保存流程，使用已有smoke检查点，不代表模型质量。报告：`runs/verification/uni_hoi_omomo_benchmark/summary.json`、`runs/verification/uni_hoi_behave_benchmark/summary.json`。真值/已知平移验证记录：`runs/verification/uni_hoi_metrics_real_data.json`。
- 论文测试划分、fps、窗口、采样预算和特征评估器尚未全面对齐，所有报告保留 `directly_comparable_to_paper_table: false`。Linux/CUDA和正式模型实验仍待运行。

硬锚点由程序写回，因此锚点误差为0只是程序正确性的检查。数步训练的样例仍存在接触距离和FK误差；没有把这些样例包装成已学会的交互。完整训练、多个种子消融、生成质量指标与物理执行成功率均尚未开展。

## 当前实现边界

已完成运动学层面的统一条件生成框架、数据/训练/采样/评估与官方代码接入。输出为22关节人体、单刚体轨迹和粗接触代理；尚无精细手指、真实力/力矩、关节限制控制器、网格穿透保证或动力学仿真闭环。

用户最终关注的“可执行HOI”还需要独立的物理跟踪与执行反馈阶段。现在的研究原型可用于建立条件生成基线、测试关系约束方案；不能称为已经实现可执行交互或已经验证新颖性的最终论文系统。
