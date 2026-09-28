# 本次交付与验证状态

交付目录：`D:/code/HOI`。用户原有 `OMOMO/` 保留。本机 `.venv/` 和 `requirements-tested.txt` 仅用于 CPU 程序检查；正式训练在 Linux 服务器另建 CUDA 环境，见 README。本次没有获得或训练好正式 HOI 模型。

## 数据实际状态

| 数据 | 已完成 | 尚需处理 |
|---|---|---|
| BEHAVE | 299条源序列已处理；293条完成真实 SMPL-H 转换，共147520帧、18种交互物体；6条因不在官方划分中而跳过；逐条数据校验通过 | 迁移到 Linux 后进行双数据集训练 |
| OMOMO | 已读取用户解压数据；全量转换并保留4736条序列、291801帧 | 源fps仍为30的显式假设，未由发布元数据确认 |
| 官方文本 | 4912条序列级标注已提取；保留数据中4669条有匹配文本 | 可选CLIP缓存未下载/运行；本次训练使用无文本条件 |

完整 OMOMO 划分为 **3797 train / 422 val / 517 test**；先按源序列分组，再生成时间窗口。测试受试者sub16/17没有进入train/val。全量清单位于 `data/processed/omomo_combined.jsonl`，默认训练配置已指向它。

原始训练来源5280条中保留4219条，拒绝1061条；原始测试602条中保留517条，拒绝85条。拒绝范围包含mop/vacuum多部件物体及超出默认1%尺度波动阈值的序列。固定中位数尺度后的完整物体表面几何改变上界，保留序列中最大约 **8.442 mm**。详情和逐条原因见两个 `conversion_report.json`，这不是无损恢复全部原始标注。

SMPL-H 不随本代码分发，下载资产仍被 Git 忽略。本地已取得官网兼容版 `smplx.zip`，原包在 `data/raw/smplh/`，男女 `.pkl` 模型在 `data/smplx_models/smplh/`。真实模型以 `num_betas=10, use_pca=False` 通过 CPU 加载及前向检查，网格形状均为 `[1,6890,3]`；下载检查见 `data/smplx_models/smplh_download_report.json`。

BEHAVE 全量转换随后在本机 CPU 完成：10fps、1024点、194 train / 17 val / 82 test。所有293条输出与原始时间戳、物体位姿逐条核对，最大骨架重建误差约 `7.46e-7 m`；接触标签与5cm采样表面距离定义一致。5条序列没有正接触代理标签，未将其伪造为有接触。完整记录见 `data/processed/behave/conversion_report.json`、`verification_report.json` 和 `verification_per_sequence.jsonl`。

联合清单 `data/processed/combined.jsonl` 含 **5029条、439321帧，3991 train / 439 val / 599 test**。源序列划分保持不变，不混入 smoke 子集。三个划分均完成全量记录加载及双数据集混合 batch 检查，文件缺失数为0；120帧窗口、60帧步进时分别产生7341 / 791 / 1397个窗口。结果见 `data/processed/combined_verification_report.json`；Linux/CUDA 与正式模型训练尚未验证。

## 已跑过的流程

- 真实 OMOMO 小样本转换：原train来源20条划为18 train / 2 val，另取官方test10条，三者没有混用。
- CPU小模型（57,753参数）训练4步：损失/梯度有限，保存模型、EMA、优化器、归一化与RNG状态；不是正式收敛实验。
- 一个检查点覆盖七种完整生成方向，以及稀疏混合H/O/C约束；检查锚点逐分量精确保留、输出有限。
- CLI生成、保存实际锚点、离线评估；无目标动作的scene-only输入也完成采样。
- 评估程序已在测试窗口上运行，并单独检查稀疏H+O模式的实际条件签名。小样本统计不构成泛化结论。
- 断点恢复测试比较连续训练与2→3步恢复，包括跨epoch情况；模型、EMA、优化器和RNG一致。
- 官方源码commit、remote、工作区洁净状态已核对；TriDi复制函数与原始源码AST相同。
- Kimodo接口以官方签名与导出字段核对，并用测试替代模型验证桥接；**没有下载或运行预训练Kimodo权重**。
- Kimodo桥接支持真实整数stride降采样、源时间戳、静态骨架反解和全源帧FK检查；HOI接入使用该人物自己的骨架尺寸。

最终单元/集成测试 **132项通过**，静态检查通过。程序报告：`runs/final_smoke/samples/report.json`、`runs/smoke_verified/benchmark/summary.json`、`runs/smoke_verified/benchmark_heldout_pairs/summary.json`。最后一次完整测试结果保存在 `runs/verification/tests.xml`。

硬锚点由程序写回，因此锚点误差为0只是程序正确性的检查。数步训练的样例仍存在接触距离和FK误差；没有把这些样例包装成已学会的交互。完整训练、多个种子消融、生成质量指标与物理执行成功率均尚未开展。

## 当前实现边界

已完成运动学层面的统一条件生成框架、数据/训练/采样/评估与官方代码接入。输出为22关节人体、单刚体轨迹和粗接触代理；尚无精细手指、真实力/力矩、关节限制控制器、网格穿透保证或动力学仿真闭环。

用户最终关注的“可执行HOI”还需要独立的物理跟踪与执行反馈阶段。现在的研究原型可用于建立条件生成基线、测试关系约束方案；不能称为已经实现可执行交互或已经验证新颖性的最终论文系统。
