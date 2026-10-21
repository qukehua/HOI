# 人体网格解码与位置一致性

原生参数预测可直接运行 `python -m unified_hoi.smpl_mesh --input PRED.npz --output MESH.npz`，按记录的SMPL-H/SMPL-X模型与手指参数完整解码。标准axis-angle参数也保存在采样NPZ中，见 [重训说明](PARAMETRIC_RETRAINING.md)。下面的22关节/FK/中性手说明仅适用于旧模型。

---

## 当前解码设置

模型继续输出 `[T,22,9]`：22 个身体关节的位置和全局旋转矩阵前两行。
网格解码通过 `R_local[j] = R_global[parent].T @ R_global[j]` 恢复局部旋转，
并将网格骨盆平移到输出的第 0 个关节。无需再次变换世界坐标系。

| 数据来源 | 人体模型 | betas | 默认资产 |
| --- | --- | --- | --- |
| OMOMO | SMPL-X | 原始 16 维 | `data/smplx_models/smplx/SMPLX_MALE.npz`、`SMPLX_FEMALE.npz` |
| BEHAVE | SMPL-H | 原始 10 维 | `data/smplx_models/smplh/SMPLH_MALE.pkl`、`SMPLH_FEMALE.pkl` |

OMOMO 设置依据固定版本的[官方加载代码](https://github.com/lijiaman/omomo_release/blob/e9d8c52f41866ed0c07ac574cb8f7f9424b6d1e7/manip/data/hand_foot_dataset.py#L237-L255)。
手指设为零旋转，与其 `run_smplx_model` 对缺失手指的处理一致；SMPL-X 的下颌、眼球和表情也保持零姿态。
本项目通过 `smplx.build_layer` 的矩阵输入接口实现，不需要额外安装 `human_body_prior`。
这不会恢复精细手指动作。

预处理保存 `body_model_type`、`body_dataset`、`gender`、`betas`，采样和 benchmark 导出时传递这些字段及 `rest_offsets`。
`body_dataset` 区分人体先验来源与物体场景的数据集；替换人体先验时清除原人物的性别和体型。
Kimodo 桥接只声明 SMPL-X 来源，不猜测其性别或原始 betas。
旧人体先验文件须用当前桥接重新转换，或显式补充其已知的人体模型来源；不能仅凭 22 关节推断。

旧处理文件可从可信的原始数据补齐人体元数据：

```powershell
.venv/Scripts/python.exe scripts/prepare_body_metadata.py --manifests data/rebuild/omomo_with_mesh.jsonl data/rebuild/behave_with_mesh.jsonl --write
```

此命令仅用于重建时的中间动作；必须显式指定 `--manifests`。当前最终 FK 数据已带完整元数据，无需执行。
原动作、接触、文本及时间数组的 ZIP 成员保持不变，只追加缺失元数据，并逐文件原子替换。
重复执行不会重复写入。与原始数据冲突的已有元数据会报错。

Python 调用：

```python
import numpy as np
from unified_hoi.smpl_mesh import record_to_smpl

with np.load("prediction.npz", allow_pickle=False) as archive:
    record = dict(archive)
vertices, faces, dominant = record_to_smpl(record)
```

命令行导出可供可视化使用的网格数组：

```powershell
.venv/Scripts/python.exe -m unified_hoi.smpl_mesh --input prediction.npz --output mesh.npz
```

上面的文件名是示例。`--models` 可指定另一台机器的模型根目录。资产被 Git 忽略，部署时需单独复制。
底层 `skeleton_to_smpl` 必须传 `dataset="omomo"/"behave"` 或显式 `model_type`；旧的仅传 22 个关节的调用须更新。
SMPL-X 网格为 10475 个顶点，SMPL-H 为 6890 个顶点，渲染器须使用函数返回的 faces，不能共用旧拓扑或固定顶点数。

有原始 betas 时直接使用，不重新拟合。缺少 betas 的旧文件可以用线性最小二乘匹配静止骨架，
但不能声称恢复原始身体表面。已知或拟合体型均检查最大骨向量误差，默认诊断阈值为 5 mm；
拟合失败会报错；已有原始 betas 则保留官方参数并警告，不自动重拟合，也可传 `strict_shape=True` 拒绝该残差。
它是资产/静止骨架一致性的诊断，不是接触精度保证。模型类型冲突和 beta 维数错误始终拒绝。
全量 4736 条 OMOMO 检查发现 sub10 的原始 betas 对应静止骨架与现有资产有最大 8.81 mm 骨向量残差，
其他受试者低于 5 mm；NPZ 和 PKL 模型的相关形状参数一致，故不能以切换这两种文件格式消除差异。
尚不能确定残差来自资产版本还是原始数据内部的静止骨架/体型不一致。保留这些差异，避免把重拟合结果误称为官方原始体型。

表面着色使用完整蒙皮权重：手指权重沿父节点汇总到同侧手腕，SMPL-X 下颌/眼球权重汇总到头部，
再在 22 个身体关节中取最大值；不会因为裁掉手指列而把手指涂成肘部或骨盆。

## 第二个问题已实施（2026-10-16）

保留 22×9 输出和 xyz 监督，统一以 `P_fk = FK(R, b, P[root])` 计算人体几何。
b 来自安装的官方人体资产与原始 betas。接触、非接触、地板、滑动损失和最终诊断均使用 P_fk；
正式训练配置开启 `losses.fk`，关系分支使用 `model.relation_geometry: fk`。无需每步运行完整蒙皮。
旧 checkpoint 未声明 relation_geometry 时保留旧分支行为，便于对照；新版训练必须使用新的 run。

### 数据与训练

已另存 4736 条 OMOMO、1454 条 BEHAVE 到 `data/processed/{dataset}_with_body_fk/`，
对应入口为 `omomo_with_body_fk.jsonl`、`behave_with_body_fk.jsonl`。
新版 NPZ 保存 `body_geometry_version=1`、模型 SHA256、原始性别/betas；
从相同根位置、全局旋转重建 xyz，并在实际物体 mesh 接触顶点上重新标记 contact。
转换时原文件不修改；清理后旧处理副本不再保留，最终文件已另存 `source_body_xyz`、`source_rest_offsets`、`source_contact` 和来源文件 SHA256。
OMOMO 源标注到新 FK 位置最大变化约 22.95 mm；这来自统一骨架，不是训练效果。

重建命令（目标清单必须尚不存在；中断后可用 `--resume` 校验并继续未完成的转换）：

```powershell
.venv/Scripts/python.exe scripts/prepare_body_geometry.py --manifest data/rebuild/omomo_with_mesh.jsonl --output data/rebuild/omomo_with_body_fk.jsonl
.venv/Scripts/python.exe scripts/prepare_body_geometry.py --manifest data/rebuild/behave_with_mesh.jsonl --output data/rebuild/behave_with_body_fk.jsonl
```

输入须先补齐原始人体元数据并关联物体 mesh。`--models` 可指定人体模型根目录。
当前两份转换报告位于 `docs/data_reports/{dataset}_with_body_fk.report.json`，包含转换数量、资产哈希和仅训练划分的整序列统计。重建命令会在新清单旁生成新的报告。
训练仍按当前训练窗口重新拟合归一化并存入 checkpoint，不使用旧统计或验证/测试划分拟合统计。
`unified.yaml`、`train_omomo.yaml`、`train_behave.yaml` 已切换入口、输出目录、FK 开关及版本要求。
`configs/ablations/` 已统一到最终 FK 数据清单，保留原网络开关与损失设置，并显式关闭文本以保持原无文本设置。数据版本变更需要新建实验，不能续训旧数据检查点。

```powershell
.venv/Scripts/python.exe -m unified_hoi.train --config configs/train_omomo.yaml
.venv/Scripts/python.exe -m unified_hoi.train --config configs/train_behave.yaml
```

上述正式训练命令默认需要 CUDA。本次仅完成 CPU 短程验证，没有启动长程训练。
新 trainer 保存 `geometry_objective_version=2`，拒绝直接 `--resume` 旧语义 checkpoint；
新版 checkpoint 支持同配置断点续训。没有实现旧权重微调迁移器。
跨机器迁移时复制新版动作目录、清单、物体 mesh、文本缓存和相同人体资产；渲染会核对资产 SHA256。

### 采样与位置锚点

采样结束始终输出 FK 位置，包括 `--projection-steps 0`。
没有非根位置锚点时无需 IK；有锚点时默认 `--ik-steps 90`，优化可动旋转和根位置。
使用递增位置罚项的 LBFGS、姿态偏离与时间平滑项；单轴伸直等零梯度情形有确定性扰动重试。
`--projection-steps N` 额外启用同一姿态求解中的接触项和可动物体修正；总迭代预算取 IK 与接触预算较大值。
`--ik-steps 0` 可关闭位置求解，但仍导出 FK 并报告残差。

已知根位置、旋转分量、物体特征及 contact 值保持固定。非根 xyz 不再硬写回：
最终解码位置在 1 mm 容差内才报告满足，否则警告并输出残差与未满足数量。
局部优化失败不等于证明约束不可行。冲突约束也不会被覆盖 xyz 隐藏。
单条采样 NPZ 保存 `human_before_fk` 和原 `*_observed`/`*_mask`，最终 `human` 才是可视化入口。
完整原始体型元数据存在时，旧 reference 会使用模型骨架重建；显式 JSON 目标和外部先验位置仍保留为请求约束。
缺少性别/betas 的外部先验只能保证给定骨架内部 FK 一致，不能声明与某个 SMPL 资产完全一致。

### 验证与评估

`human_fk_error_max_m` 衡量最终位置与 FK 的最大偏差；
`human_decoded_anchor_{mean_m,max_m,unsatisfied_count,status}` 报告实际解码锚点。
接触、滑动、地板和人体运动统计全部从 FK 计算，原始 feature 误差单独保留。
离线评估优先用预测文件保存的 rest_offsets，避免用旧 reference 骨架重新解释新预测。
OMOMO 的 24 关节指标用实际 SMPL-X 骨架及中指根计算新版预测，原始24关节标注仍作为评价参照；
报告注明这一解码区别，不宣称论文表格数值直接可比。

历史回归检查覆盖几何梯度、伪造 xyz、可行/冲突/部分锚点、padding、
源数据保留、训练统计隔离、旧续训拒绝，以及四个真实男女 SMPL-X/SMPL-H 模型的 FK/网格一致性。
两数据集曾完成完整网络 CPU 短训练、采样、网格导出、离线诊断和 benchmark；
对应测试代码和产物于 2026-10-16 按用户要求列入清理清单，实际执行状态见 `docs/project_cleanup_report.json`。这些历史检查验证代码链路，不代表模型已经收敛。

手腕中心接近物体仍是粗接触代理。掌面贴合、手指抓握及无穿透需要另外引入手部姿态与网格接触项。
