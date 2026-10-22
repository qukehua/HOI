# 原生人体参数、128维接触与隔离重训

当前在 `D:/code/HOI` 使用原生人体参数与接触latent模型，从头训练。`data/` 已清理为当前模型依赖，目录用途见 [data/README.md](../data/README.md)。旧预处理数据、原始BEHAVE和原始文本标注已按用户明确选择删除。正式HOI训练由用户在GPU服务器启动。本机的两步训练只验证程序，不代表模型质量。

## 输出与评估

| 项目 | BEHAVE | OMOMO |
| --- | --- | --- |
| 原生模型 | SMPL-H，10 betas | SMPL-X，16 betas |
| H维度 | 325 | 331 |
| 局部姿态 | 根+21身体+左右各15手指关节，52×6D | 同52个身体/手关节，SMPL-X额外下颌/眼球与表情固定中性 |
| 平移 | 原生SMPL transl，3维 | 原生SMPL transl，3维 |
| O | 刚体原点平移3+旋转6D | 同左，保留序列物体尺度 |
| C | 128维连续latent | 独立的128维连续latent |
| 接触解码 | 本数据集AE → 690个概率 | 本数据集AE → 690个概率 |
| 评估关节 | SMPL-H原生73关节 | body22+hand30+vertex landmarks21，共73个同语义关节 |
| 接触点 | TriDi官方690个SMPL-H顶点索引 | 原生SMPL-X上独立选择690个点，不复用SMPL-H索引 |

H的顺序是 `[betas, local_rot6d[52], transl]`，旋转6D存矩阵前两**行**。OMOMO的SMPL-X 127输出中，选择 `0:22`、`25:55`、`55:76`；这不是简单取前73个。SMPL-H的6890顶点与SMPL-X的10475顶点不会互相套索引。

采样NPZ同时保存 `global_orient[T,3]`、`body_pose[T,63]`、`left_hand_pose[T,45]`、`right_hand_pose[T,45]`、`betas[T,N]`、`transl[T,3]`，均为标准axis-angle姿态参数；使用 `use_pca=False, flat_hand_mean=True` 解码。另有 `human_params`、`local_pose6d`、`joints73`、`contact_latent`、`contact_probability`、`contact_vertices`、条件mask、时间戳和资产校验信息。OMOMO还保存中性的表情/下颌/眼球参数。

当前源数据的限制：逐一检查BEHAVE的299份原始SMPL-H拟合，手指轴角在序列内及跨序列均是同一固定值，转换保留该值。OMOMO发布数据没有手指姿态，使用中性手姿并在扩散中作为固定约定。**现有数据不能监督真实动态手指运动**；增加输出维度不能补出不存在的标注。未来若数据含动态手指，BEHAVE转换器会拒绝丢弃它们，需接入按时间戳读取的完整标注。

## 网络和训练改动

`src/unified_hoi/parametric/` 是当前实现；公共入口仅调用这一路径，旧body-FK实现和配置已在上一轮清理中删除。默认仍为width192、4个时空分解block和粗阶段，BEHAVE可训练参数7,380,420，另有冻结AE参数1,036,594。每帧52姿态token、shape、transl、object、contact共56个token。没有用“单纯加深网络”混淆这次输出及评估修正。

物体条件沿用网格1024点BPS。关系分支从当前带噪的局部姿态和betas执行FK，得到22个身体关节与物体的关系，按物体噪声程度门控，并使用物体query聚合。**关系分支的22关节不是最终评估关节数**。生成人体时不从GT骨架或GT betas补入形状。

训练默认使用 `timestep_sampling: shared`：每个样本随机抽一个时间步，所有未知模态共享它，完全已知模态使用0；不同样本仍独立抽样，各模态仍使用独立高斯噪声。101为`(t,0,t)`，011为`(0,t,t)`，111为`(t,t,t)`，与当前同步去噪的推理路径一致。OMOMO固定手指属于部分已知条件，不会把整个人体误判为已知。可显式设为`independent`进行旧策略对照。

共享时间步也会改变现有几何监督的触发频率：默认1000步、`geometry_max_t: 250`时，110/111的人体和物体同时处于低噪声区间的样本概率从6.25%变为25%。几何损失公式、权重和帧数上限不变，实际参与计算的帧仍受`geometry_frames`限制；分析消融结果时需计入这个联动影响。

未知模态生成，已知模态逐步原样回填。局部姿态、betas、平移和128维接触latent的去噪目标保持不变；观察项系数0.05保留在分子，不被分母抵消。低噪声阶段使用可微原生LBS的690个表面顶点计算身体误差，并计算物体对应点误差及预测/GT表面距离误差。当前接触距离监督使用完整三角网格中被面引用的全部物体顶点和序列尺度，与已存接触标签的生成方式一致；物体V2V训练及关系分支仍使用512个固定点。距离均为顶点最近距离，不是三角面有符号距离或穿透量。

训练只用train拟合归一化与AE。AE结构是690→512→256→128→256→512→690，与TriDi的层尺寸一致；使用加权BCE，独立训练3000步，val选检查点和概率阈值。官方接触权重下载服务在本次访问返回503，因此这里没有加载官方权重，也不声称两个latent空间相同。两套AE训练得到不同坐标系，严禁互换。

验证在val中均匀抽取固定窗口，默认8个、DDIM50步；新配置按1fps评分且不使用文本，与默认测试协议一致。详见 [统一评估](EVALUATION.md)：

- `best_010.pt`：按mode010的OBJ_V2V选择。
- `best_100.pt`：按mode100的MPJPE选择。
- `best.pt`：由 `best_mode` 指定，默认010。
- `last.pt`：含模型、EMA、优化器、RNG、epoch和batch位置，用于续训。

如果论文实验要求同一个检查点覆盖全部模式，预先固定 `best_mode` 并对全部模式使用同一 `best.pt`；不能根据test结果挑选检查点。更稳健的实验可在开始前增大 `validation_batches`，固定采样步数和验证预算。

当前支持七种完整H/O/C模式；旧XYZ关键帧IK、任意关节分量控制、稀疏接触mask和接触投影没有移植到latent。公开CLI已移除这些旧选项，传入会报错。

## 接触监督（第三项）

BEHAVE、OMOMO和默认训练配置启用以下三个可独立消融的选项：

```yaml
contact_supervision:
  decoded_bce_weight: 1.0
  balance_surface_distance: true
  full_mesh_distance: true
```

1. 保留latent MSE，新增`contact_bce`：预测的干净接触latent先反归一化，再经本数据集的冻结AE decoder得到690个logits，与GT标签计算`binary_cross_entropy_with_logits`。只有C未知且有效的帧参与；已知接触和padding不参与。使用所有C噪声时间步的x0预测，不额外施加H/O低噪声门限。AE权重保持冻结，解码过程保留梯度，使BCE能训练接触生成分支。
2. BCE和`surface_distance`分别计算GT正接触点、负接触点的平均误差，两类都有时各占一半；只有一类时使用该类均值，支持集为空时返回零。几何距离只统计H或O未知的所选帧，继续匹配GT距离。这样大量非接触点不会仅凭数量主导损失，也不会强迫每个正接触点达到零距离。
3. `full_mesh_distance`让预测和GT的距离都使用清单引用的完整三角网格的全部面引用顶点，保留每个序列的尺度；不再用512点近似接触距离。不同物体和不同尺度分组计算，大网格分块寻找最近点，反向图只保留选中的近邻。计算量会增加，仍受`geometry_frames`和原有H/O低噪声门限限制。

几何口径的已知历史差异：原始预处理使用`vertices[np.unique(faces)]`，当前`tridi_metrics_v2`几何评估使用资产中的全部顶点（含孤立顶点）。BEHAVE首个训练窗口的8280个标签中发现1个因此不同，不能将两个口径声称为完全等价。本项采用与现有标签、AE一致的完整面引用顶点；没有修改数据、AE或已有评估协议。若以后统一孤立顶点处理，需对双方结果使用同一新版协议重新评分。

总损失在原有项上增加`decoded_bce_weight * contact_bce`；`geometry_losses.surface_distance`继续控制几何接触项的权重。默认BCE权重1.0是实验起点，尚无正式重训结果证明它最优。人体分组损失、额外关节损失和关系门控没有随本项修改，也未加入两个预测之间的自一致性损失。

日志新增`contact_bce`、`contact_bce_weighted`、`contact_decoded_positive/negative`、正负点计数及`contact_decoded_precision/recall/f1`。这些是训练时一次x0预测的统计，使用本数据集AE选定的阈值，**不是验证集完整采样指标**。`contact_decoded_frames=0`时这些解码统计为零，不能作为接触质量结论。`contact_distance_positive/negative`及计数记录实际选入几何监督的两类误差。

恢复旧接触目标、只测共享时间步时，将以上三个选项分别设为`0.0`、`false`、`false`。旧配置缺少`contact_supervision`也按这组关闭值解释；当前发布的三个训练YAML显式启用新监督。配置、`run_info.json`及检查点记录该设置，改变目标需新建训练实验，不能直接`--resume`旧目标。已有训练数据和两套AE无需重做；旧HOI检查点仍可按原方式采样、评估。

## 服务器需要的文件

在服务器项目根目录建立CUDA环境并安装 `pip install -e '.[body,swanlab]'`；PyTorch的CUDA版本按服务器驱动选择。不要复制Windows `.venv`。下列数据和权重被Git忽略，必须另外迁移：

| 文件/目录 | 用途 |
| --- | --- |
| `data/processed/behave_parametric.jsonl` 与 `behave_parametric/` | BEHAVE原生训练数据 |
| `data/processed/omomo_parametric.jsonl` 与 `omomo_parametric/` | OMOMO原生训练数据 |
| `data/processed/object_meshes/` | 清单引用的物体网格 |
| `data/processed/text_features/` | 清单引用的真实CLIP文本缓存 |
| `data/parametric_assets/behave/` | contact_support.npz与contact_ae/best.pt |
| `data/parametric_assets/omomo/` | 独立的同类资产 |
| `data/smplx_models/smplh/`、`data/smplx_models/smplx/` | 已授权的原生男女模型 |

本机新清单中的数据、网格、文本路径均相对清单目录，保持目录结构即可。`source_path`是来源说明，不用于训练加载。SMPL系数不打包进HOI检查点，恢复/采样仍需相同模型文件。数据、接触点和检查点均检查dataset与SHA256；混用会报错。

当前原生训练数据和AE已经准备完成，迁移整个 `data/` 即可，无需再次转换。旧 `*_with_body_fk` 中间数据、原始BEHAVE及标注已删除，旧转换工具也已删除；若将来需要从原始材料重建，必须重新取得源数据并从Git历史恢复对应预处理工具，不能仅凭现在的训练文件重新生成所有旧文件。

```bash
# 验证已有清单与codec；当前数据已全部准备好。
python scripts/prepare_parametric.py --dataset behave
python scripts/prepare_parametric.py --dataset omomo

# 独立正式训练，共享时间步+新接触监督；默认分别200k和500k步，CUDA:0。
python -m unified_hoi.train --config configs/train_behave.yaml --output runs/behave_shared_t_contact_seed42
python -m unified_hoi.train --config configs/train_omomo.yaml --output runs/omomo_shared_t_contact_seed42

# 续训必须使用同一数据、网络和训练语义；可以增加max_steps。
python -m unified_hoi.train --config configs/train_behave.yaml \
  --output runs/behave_shared_t_contact_seed42 --resume runs/behave_shared_t_contact_seed42/last.pt
```

可加 `--no-swanlab` 关闭日志服务。新实验使用新的 `--output`。旧22关节检查点不能作为新模型 `--resume` 的输入。

`config.json`、`run_info.json`和检查点配置记录时间步策略。旧检查点缺少`timestep_sampling`时按`independent`识别，不能直接作为`shared`实验的`--resume`；原实验续训需保留原配置并显式设置`independent`，其他配置仍需一致。共享时间步本身没有修改关系门控、损失权重或模型结构；新增接触监督由上面的独立配置控制。

## 两套数据使用相同重建指标

模式位依次是H/O/C，1生成、0条件：010生成物体、100生成人体；011/101还生成接触。

- `MPJPE`：原生解码73关节，分别以骨盆对齐后的平均欧氏距离，米。
- `MPJPE_PA`：再做旋转/平移/尺度Procrustes对齐，米。
- `OBJ_V2V`：完整刚体模板对应顶点平均距离，米。
- `OBJ_CENTER`：完整模板顶点几何中心的距离，米。
- `OBJ_CONTACT_MESHES` / `SBJ_CONTACT_MESHES`：690身体表面点到完整物体顶点集，≤5cm为接触，预测和GT标签一致率，百分数。
- `*_CONTACT_DIFFUSED`：C生成时通过冻结AE解码并阈值化后与GT表面接触比较；C已知时沿用实际GT接触，符合TriDi条件评估语义。

额外报告precision/recall/F1，避免大量非接触点使准确率显得很高。`CONTACT_MESHES`必须来自生成的人/物几何，不能用接触解码器输出冒充。`CONTACT_DIFFUSED`才评估latent恢复的接触。

```bash
python scripts/benchmark.py \
  --checkpoint runs/behave_parametric_seed42/best.pt \
  --manifest data/processed/behave_parametric.jsonl \
  --profile tridi --modes 010 100 --window 120 --steps 50 \
  --device cuda:0 --save-predictions --output runs/eval_behave_parametric

python scripts/benchmark.py \
  --checkpoint runs/omomo_parametric_seed42/best.pt \
  --manifest data/processed/omomo_parametric.jsonl \
  --profile tridi --modes 010 100 --window 120 --steps 50 \
  --device cuda:0 --save-predictions --output runs/eval_omomo_parametric
```

上述是单样本、无文本条件的010/100；需要文本时显式加 `--text-conditioned`，缺失缓存会报错。生成接触用 `--modes 011 101`，只与baseline同模式比较。`--profile tridi`默认1fps评分，但保持原始序列帧率生成。主结果在`summary.json.reconstruction`，使用逐帧平均和全局接触F1；窗口平均仅作诊断。分别使用best_010/best_100及缓存复评的命令见 [统一评估说明](EVALUATION.md)。

**这些命令对齐的是指标公式和输出语义，不是完整论文表格复现。** 当前仍采用本项目的片段划分、窗口、过滤过的刚体对象和独立接触AE；OMOMO原始fps假设仍保留旧流水线限制。现已接入1-NNA/COV/MMD/SD，其中MMD/SD按TriDi默认FAISS的平方距离计算。报告中仍保留`directly_comparable_to_paper_table=false`，在统一split、实际源帧、坐标预处理、采样预算及模型选择协议前，不直接与原论文数字比较。

## 单条采样、重新评估和完整网格

将 `REF.npz` 替换为对应原生清单中的实际文件路径；两套数据使用各自检查点/清单。

```bash
python -m unified_hoi.sample \
  --checkpoint runs/behave_parametric_seed42/best.pt \
  --manifest data/processed/behave_parametric.jsonl --reference REF.npz \
  --mode 101 --frames 120 --steps 50 --device cuda:0 --output runs/sample_native.npz

python -m unified_hoi.evaluate \
  --prediction runs/sample_native.npz --reference REF.npz \
  --checkpoint runs/behave_parametric_seed42/best.pt \
  --manifest data/processed/behave_parametric.jsonl --output runs/sample_native_metrics.json

python -m unified_hoi.smpl_mesh \
  --input runs/sample_native.npz --output runs/sample_native_mesh.npz
```

复评检查保存的mode、窗口、时间戳、身体资产、接触support/codec和已知条件是否一致，不自动对齐错误帧。完整网格输出BEHAVE6890或OMOMO10475顶点。

## 当前资产与验证边界

现成原生数据包含1454条BEHAVE和4736条OMOMO记录。两套接触AE分别训练，所选步数、阈值及验证F1保存在各自 `contact_ae/best.pt` 元信息中；AE重建质量不等于从噪声生成HOI的质量。

此前CPU短程验证覆盖训练、010/100/001采样、结果复评、完整网格导出及断点续训。这些记录只证明流程可运行，正式模型效果仍需服务器训练后判断。按用户要求，专用测试代码及历史报告列入清理清单，执行状态统一见 `cleanup_tests_docs_20261022.json`。
