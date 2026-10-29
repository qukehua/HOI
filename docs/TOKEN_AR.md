# Token-AR：masked autoregressive VQ 的 HOI 模型

项目只保留本模型（`src/unified_hoi/token_ar/`，架构标识 `token_ar_v1`，配置在 `configs/token_ar/`）。原连续扩散模型已于2026-10-29删除，可从 git 历史恢复；扩散检查点与本模型不兼容。

**正式模型：`Qwen/Qwen3.5-4B-Base`，全参数训练 + 七种 H/O/C mask，不使用 LoRA（2026-10-22）。** 两套数据的配置为 `configs/token_ar/qwen35_4b_behave.yaml` 和 `qwen35_4b_omomo.yaml`，配套四卡配置为 `accelerate_qwen35_4gpu.yaml`。Qwen3-4B/8B 配置保留作对照。

身体解码、数据读取、评估协议和指标函数最初从扩散模型复制后修改，来源及原文件 SHA256 记录在 `TOKEN_AR_ORIGIN.json`。`src/unified_hoi/` 顶层的清单、旋转、网格 BPS、SMPL 网格和运行工具由本模型直接使用。

## 新模型的数据流

1. H：沿用 BEHAVE 325维／OMOMO 331维原生参数，独立人体 VQ-VAE。
2. O：沿用平移3＋旋转6D，共9维，独立物体运动 VQ-VAE。
3. C：690个表面接触标签，独立接触 VQ-VAE；不加载旧接触 AE 权重。
4. G：网格经过现有1024锚点 BPS，分为8组，由可训练投影映射成8个连续几何前缀 token。
5. T：读取清单中的原始文本，用基础语言模型 tokenizer 编码，不读取 CLIP embedding。

每个 VQ-VAE 默认512个码、128维 latent、时间压缩倍率2。三套码本互不共享，BEHAVE／OMOMO也不共享。只在 train 上拟合归一化和训练码本，val 选择检查点。人体／物体使用标准化重建、速度、码本及 commitment 损失；接触使用按正负类别平衡的 BCE、码本及 commitment 损失。记录 perplexity，并可重置近期未用的码；这属于待验证的初始设置，不是论文结果复现。

AR 输入：几何前缀、文本与帧数/fps/性别信息、任务 mask、已知模态的完整 token 序列、未知模态的 missing 标记。目标按时间交错排列，例如111模式为 `H0 O0 C0 H1 O1 C1 ... END`。每次只预测一个 token，随后通过相应 VQ-VAE 解码。长度由请求的帧数确定，不依赖提前生成 END；采样时按当前位置限制到对应模态码本。

采用预训练语言模型的 causal backbone 和原始文本 embedding，新建运动 embedding 与运动输出 head。输出 head 只预测 HOI 词表，避免为每个运动 token 计算完整语言词表 logits。文本当前只作为输入，不做动作描述生成。物体几何始终是条件，不生成新网格。

## mask 与参数更新范围

mask 控制任务，不是参数更新算法。新分支使用正常因果注意力与 teacher forcing，源条件和 padding 的 label 为 `-100`，只计算缺失模态目标和 END 的 token 交叉熵。不是 BERT 式双向填空，也不是扩散。

| mask（H/O/C，1生成、0已知） | 生成内容 | 已知条件（另有mesh和可选文本） |
| --- | --- | --- |
| 100 | H | O、C |
| 010 | O | H、C |
| 001 | C | H、O |
| 110 | H、O | C |
| 101 | H、C | O |
| 011 | O、C | H |
| 111 | H、O、C | 无额外运动／接触条件 |

训练按样本随机选择模式，一套共享权重覆盖七种模式。采样只编码已知模态，绝不把未知 GT 的 token 放进 prefix。解码后，已知人体、物体、接触直接保留原始数值，不经过有损重建。只支持完整模态 mask；当前没有稀疏帧、任意关节、局部接触编辑接口。

`train_scope` 有三种取值，均不使用 LoRA：

- `full`：更新整个语言 backbone，加上几何投影和运动 embedding/head。
- `last_layers`：只解冻 backbone 最后 `last_layers` 个 block 和末尾 norm；几何投影和运动 embedding/head继续更新。
- `frozen`：冻结 backbone，只训练新接入模块，适合低预算对照，不能假设效果等同完整微调。

选定的 Qwen3.5-4B 预设为 `full`：32层语言骨干、原始文本 embedding、几何投影及新运动 embedding/head 全部更新。VQ-VAE 在第一阶段独立训练，第二阶段固定码本，避免运动 token 的含义漂移。旧 Qwen3-4B/8B 对照配置分别为全参数、末4层解冻；不按模式保存不同 adapter。

## Qwen3.5 接入

官方发布的是多模态权重包，本模型仅取 `Qwen3_5TextModel` 语言骨干。几何仍走 BPS 投影；不实例化视觉编码器、原语言输出 head 或辅助 MTP 模块。`model.language_model.*` 权重显式映射到文本骨干；缺失任何骨干权重会直接报错，避免误用随机初始化参数。第一次下载可能仍涉及含其他模块的共享权重分片。

两份正式配置固定官方 revision `1001bb4d826a52d1f399e183466143f4da7b741b`，使 tokenizer、配置和权重来自同一版本。语言骨干参数精确计数为4,205,751,296；结构为隐藏维度2560、32层，其中24层 Gated DeltaNet 线性注意力、8层全注意力。输入、输出仍是因果 token 序列。推理采用 Transformers 的混合缓存，保存检查点时记录纯文本骨干配置，可离线重建。

四卡 FSDP 包装层改为 `Qwen3_5DecoderLayer`；不要将 Qwen3.5 配置与旧 `accelerate_4gpu.yaml` 混用。使用 FP32 主权重、BF16 混合精度、梯度检查点、每卡 batch=1、累积8次、512 token 上限。`sync_each_batch: true` 在每个微批次同步并分片梯度，避免 FSDP 的 `no_sync` 在累积期间保留完整梯度；优化器仍按累计步数更新。该设置以额外通信换取显存余量。

## 基础模型候选与4×32GB显存

下列参数规模和许可证来自作者模型卡；显存建议为工程估算，不是本项目测速。

| 候选 | 许可证 | BF16权重约占用 | 建议用途 |
| --- | --- | --- | --- |
| [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | Apache-2.0 | 语言骨干约8.4GB；完整多模态包更大 | **已选定：四卡FSDP全参数 + mask** |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | Apache-2.0 | 3.4GB | 较便宜的全参数消融与调试 |
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Apache-2.0 | 8GB | 保留旧骨干对照 |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Apache-2.0 | 16GB | 对齐论文骨干；当前四卡先采用末4层解冻 |
| [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | MIT | 约7.6GB（3.8B） | 不同模型家族对照，需另验 tokenizer、训练和FSDP层类型 |

Qwen3-4B/8B已提供配置；HF backbone接口可以更换模型，但不是所有 `AutoModel` 架构都经过验证。更换到Phi等架构时必须更新FSDP的 `fsdp_transformer_layer_cls_to_wrap`，不能直接沿用 `Qwen3DecoderLayer`。

8B单卡BF16推理的权重约16GB，但还需要KV cache、激活和临时空间，24–32GB卡可作为短序列起点。全参数AdamW训练按16–18字节/参数估算，模型状态合计约128–144GB，另加激活和通信缓冲；4×32GB比较紧，4×48GB更有余量。四张卡的显存不会自动合并：FSDP分片才会分摊参数、梯度和优化器状态。参考 [HF显存说明](https://huggingface.co/docs/transformers/main/en/model_memory_anatomy) 和 [FSDP说明](https://huggingface.co/docs/accelerate/usage_guides/fsdp)。

新代码使用FP32主权重配合BF16 autocast，4B默认小批量、梯度累积、梯度检查点、SDPA、512 token上限和FULL_SHARD。每个rank从CPU加载模型；完整检查点保存还需要主机RAM和磁盘空间，尤其是8B。不要把推理显存估算当训练显存保证。

## 服务器训练

在项目根目录运行，先配置支持RTX5090/Blackwell的PyTorch CUDA环境。[PyTorch 2.7起提供CUDA12.8/Blackwell支持](https://pytorch.org/blog/pytorch-2-7/)，应选与你服务器驱动兼容的当前版本。本机验证使用CPU，未验证CUDA、NCCL、四卡FSDP吞吐或实际峰值显存。四卡配置面向Linux环境。

```bash
pip install -e '.[body,token-ar]'

# CUDA/Linux 上可安装 Qwen3.5 优化内核；缺少时 Transformers 使用较慢的 PyTorch 实现。
# 需先配置匹配 PyTorch 的 CUDA 编译环境；本机 CPU 验证未测试这些内核。
pip install flash-linear-attention causal-conv1d --no-build-isolation

# 第一阶段：独立训练H/O/C的VQ-VAE；此入口目前用单GPU。
python -m unified_hoi.token_ar.train_codecs --config configs/token_ar/vq_omomo.yaml
python -m unified_hoi.token_ar.train_codecs --config configs/token_ar/vq_behave.yaml

# 第二阶段：已选定 Qwen3.5-4B-Base，全参数训练，同一套mask多任务权重。
accelerate launch --config_file configs/token_ar/accelerate_qwen35_4gpu.yaml \
  --module unified_hoi.token_ar.train --config configs/token_ar/qwen35_4b_omomo.yaml

# 可选8B候选：末4层解冻，仍为同一套mask多任务权重。
accelerate launch --config_file configs/token_ar/accelerate_4gpu.yaml \
  --module unified_hoi.token_ar.train --config configs/token_ar/qwen3_8b_omomo.yaml
```

BEHAVE把配置后缀换成 `behave`。两个数据集分别训练，不要交换码本。正式步数是实验起点，需根据VQ重建及验证表现调整。现有原生数据无需重新转换；原生人体模型、contact support和对象网格仍需随 `data/` 迁移。

新运行要求输出目录为空；续训需要显式提供检查点。VQ使用 `last.pt`／`best.pt`；AR保存带 `COMPLETE` 标记的 `checkpoint-0001000/` 等目录，其中包括共享模型权重、优化器/RNG状态、tokenizer、VQ检查点和架构/数据校验信息。

2026-10-29：按用户选择，BEHAVE正式配置 `qwen35_4b_behave.yaml` 的新AR实验使用VQ训练完成后保存的 `runs/token_ar/behave/vq/last.pt`。`best.pt` 仍是按原总验证损失保存的诊断记录，不作为这份配置的码本来源。固定训练预算后评估最终权重的重建质量，记录其step与SHA256；启动AR后不要继续覆盖其来源VQ文件。更换码本需要新的AR实验和新的输出目录，不能接着旧AR检查点续训；已有AR结果的复评继续使用检查点内绑定的 `codecs.pt`。

```bash
python -m unified_hoi.token_ar.train_codecs --config configs/token_ar/vq_omomo.yaml \
  --resume runs/token_ar/omomo/vq/last.pt

accelerate launch --config_file configs/token_ar/accelerate_qwen35_4gpu.yaml \
  --module unified_hoi.token_ar.train --config configs/token_ar/qwen35_4b_omomo.yaml \
  --resume runs/token_ar/omomo/qwen35_4b_full/checkpoint-0001000
```

AR续训保持配置和进程数一致，可提高max_steps；VQ码本或清单改变会拒绝续训。`best.json`指向按七模式验证token CE的等权平均选出的单一检查点。这里的验证是teacher-forced预测，不是完整生成的MPJPE、接触F1或物理质量。不能用test挑检查点。

## P0：三套 VQ 的重建基线

无需重新训练，也不加载 Qwen。`evaluate_codecs` 直接将验证集真实 H/O/C 编码成各自的离散 token，再用冻结的 VQ 解码。默认一次评估100（仅重建人体）、010（仅重建物体）、001（仅重建接触）和111（联合重建）；mask为0的模态保留真实值。这是有真实目标token的重建诊断，不是条件生成成绩，也不是数学上的性能上界。

诊断已有 AR 结果时，优先使用**该 AR 检查点自带的 `codecs.pt`**，避免之后更新的VQ权重与原实验码本不同。一次加载的文件已经包含三套独立VQ，无需分别启动三次任务。

在服务器项目根目录运行（Windows使用 `.venv/Scripts/python.exe`）：

```bash
.venv/bin/python -m unified_hoi.token_ar.evaluate_codecs \
  --checkpoint runs/behave_10.27/checkpoint-0001000/codecs.pt \
  --manifest data/processed/behave_parametric.jsonl \
  --contact-support data/parametric_assets/behave/contact_support.npz \
  --split val --window 120 --eval-fps 1 --device cuda:0 \
  --output runs/eval_behave_vq_oracle_step1000
```

若只测试第一阶段最终模型，改为 `--checkpoint runs/token_ar/behave/vq/last.pt`。输出目录必须是新目录。冒烟检查可加 `--limit 2` 并另用一个输出目录；正式结论需要不加limit的完整val。默认评估1fps、73关节、690接触顶点、5cm几何接触阈值，沿用AR的 `tridi_metrics_v2` 公式；VQ仍编码原始帧率的完整有效窗口，再抽帧计算指标。

输出 `summary.json` 和 `per_clip.jsonl`，重点读取：

| JSON位置 | 含义 |
| --- | --- |
| `reconstruction.100.MPJPE` / `MPJPE_PA` | 人体重建关节误差；MPJPE相对骨盆 |
| `reconstruction.100.HUMAN_TRANSL_L2` / `HUMAN_V2V_690` | 世界坐标SMPL平移误差／690个接触顶点重建误差，补充根相对MPJPE未覆盖的信息 |
| `reconstruction.010.OBJ_V2V` / `OBJ_CENTER` / `OBJ_ROTATION_DEG` | 物体顶点／中心／旋转重建误差 |
| `reconstruction.001.CONTACT_TOKEN_F1` | 接触VQ解码与GT网格接触比较，可与AR的接触token F1对照 |
| `contact_diagnostics.VQ_VS_NATIVE` | 接触VQ对其训练用原始二值标签的P/R/F1 |
| `contact_diagnostics.NATIVE_VS_GT_MESH` | 原始接触标签与评估重算的GT网格接触差异，用于排查标签口径 |
| `reconstruction.100.SBJ_CONTACT_F1` / `010.OBJ_CONTACT_F1` / `111.SBJ_CONTACT_F1` | 单独重建H、单独重建O、联合重建后，实际几何接触是否被保留 |
| `token_usage.human` / `object` / `contact` | 全部有效token的使用码数、使用比例、perplexity与直方图，排除padding |

距离单位为米（乘100得到cm），旋转为度，P/R/F1为0–1。接触主指标按全局TP/FP/FN汇总，读取 `reconstruction` 或 `contact_diagnostics`，不要用逐窗口F1的平均值代替。100/010中的接触token是已知GT，不能用它们证明接触VQ重建良好。旧报告中的 `CONTACT_DIFFUSED_F1` 若表示解码接触，与此处 `CONTACT_TOKEN_F1` 对应；`CONTACT_MESHES` 原始列是accuracy%，应比较另列的 `SBJ_CONTACT_F1` / `OBJ_CONTACT_F1`。

判断方法：在相同VQ权重、划分、窗口和评估帧上，如果VQ重建已经接近当前AR的较大误差，先改善第一阶段表示与重建；如果VQ重建明显更好，则主要排查AR预测与条件利用。H/O各自误差较小但111网格接触F1明显下降，说明独立量化可能破坏交互一致性，连续几何修正值得验证。低码本利用率只是一项诊断，不能单凭它认定坍塌。新报告可用 `evaluation_frames_sha256` 与 `reference_values_sha256` 核对同一评估对象；旧JSON若缺少这些字段，只能先对照划分、帧数和协议。

## 采样与评估

下例把 `REF.npz` 替换为清单里的真实原生记录，检查点替换为已训练目录。

```bash
python -m unified_hoi.token_ar.sample \
  --checkpoint runs/token_ar/omomo/qwen35_4b_full/checkpoint-0001000 \
  --manifest data/processed/omomo_parametric.jsonl --reference REF.npz \
  --mode 101 --frames 120 --text 'Lift the box and put it down.' \
  --device cuda:0 --dtype bfloat16 --output runs/token_ar/sample_101.npz

python -m unified_hoi.token_ar.evaluate \
  --checkpoint runs/token_ar/omomo/qwen35_4b_full/checkpoint-0001000 \
  --manifest data/processed/omomo_parametric.jsonl --reference REF.npz \
  --prediction runs/token_ar/sample_101.npz --output runs/token_ar/sample_101_metrics.json
```

`--text-conditioned`使用清单caption，`--text`使用自定义prompt；两者都不指定则关闭文本。默认temperature=1、top-k=50；temperature=0为贪心。采样支持固定帧数，不是未来帧预测专用接口。默认评估1fps，保留现有73关节和几何接触指标公式；`--eval-fps 0`评估全部有效帧。`CONTACT_TOKEN`指接触token解码，`CONTACT_MESHES`来自人体和物体几何，两者独立报告。

保存原生人体参数、标准axis-angle姿态、物体姿态、接触概率、三路token、条件mask、时间戳和资产信息。SMPL-X手指仍固定中性；现有数据没有动态手指监督。

批量基准按 `tridi_metrics_v2` 口径，对 val/test 全部窗口逐模式采样并评分：重建指标（MPJPE、PA-MPJPE、V2V、物体中心/旋转误差、接触 P/R/F1，帧加权、接触按全局 micro 汇总）以及生成指标 1-NNA/COV/MMD/SD（平方 L2，人体特征相对骨盆，SD 的训练参考默认10fps）。

```bash
python -m unified_hoi.token_ar.benchmark \
  --checkpoint runs/behave_10.27/checkpoint-0001000 \
  --split val --modes 100 010 001 111 --temperature 0 \
  --device cuda:0 --dtype bfloat16 --output runs/eval_behave_10.27_step1000
```

输出 `summary.json`（含协议版本、评估帧和参考集 SHA256，便于判断两次结果是否同口径）与逐条 `per_clip.jsonl`。`--limit N` 只评前N个窗口用于冒烟检查；`--no-text` 关闭清单文本；`--profile native` 评估全部帧。不能用 test 选检查点。

## 目标与已验证边界

AR训练使用离散目标token CE；接触BCE在接触VQ-VAE训练阶段使用，不是AR阶段的可微几何约束。2026-10-29新增可选的连续修正阶段，见下文。它的几何梯度只更新修正网络，不穿过离散采样回传到Qwen。尚无正式训练后的质量结果，也没有带符号穿透损失，不能声称已保证物理一致性。

已有CPU验证使用随机初始化的微型Qwen3/Qwen3.5和真实数据小窗口，覆盖两套数据的VQ/AR训练与续训、010/100/001/111采样、复评及完整网格解码。历史Qwen3.5验证记录为38项单元测试通过，并检查了条件保持、未知GT隔离、权重映射、全参数更新和缓存一致性。

2026-10-27起已在4×RTX5090上用正式4B权重训练BEHAVE（`runs/behave_10.27`）。训练CE趋近0而验证CE在1000步最低，之后持续上升，属于明显过拟合（训练窗口仅约1745个）；正式结论需以批量基准结果为准。Qwen3.5优化内核（causal-conv1d、flash-linear-attention）尚未安装。

## 可选第三阶段：连续几何修正（2026-10-29）

`refinement.py` 实现一个共享七种mask的时序残差网络。输入为真实AR采样并经VQ解码的H/O/C、mask、fps、物体BPS，以及由这些粗结果计算的人体22关节物体局部坐标、最近网格顶点向量和距离。forward不接收未知模态GT。网络修正生成模态的人体根平移、52个局部关节旋转、物体旋转/平移及690点接触logit。体型沿用粗生成值；OMOMO中性手约定保留。旋转通过SO(3)残差更新，已知模态逐值保持原样。

训练使用独立缓存，Qwen和三套VQ全部冻结。缓存必须来自自由自回归生成，不使用teacher forcing或GT token重建冒充生成误差。缓存制作需要运行4B，但修正网络训练本身不加载Qwen，使用单卡FP32。缓存仅包含train/val，禁止用test训练或选模。验证遍历缓存里的全部val窗口、全部选定模式；物理损失每窗均匀选择最多16帧，不等同于完整benchmark。

监督项：人体690个选定网格顶点MSE、物体全网格顶点MSE（用网格矩精确计算）、人体—物体相对关节位置、平衡的表面距离误差、解码接触BCE、几何接触BCE、接触—几何双向一致性、旋转误差、GT速度匹配、残差正则。最近顶点索引查找不求导，但距离对人体顶点和物体位姿可微。接触距离沿用评估的最近网格顶点代理，阈值5cm；不是带符号三角面距离或严格穿透约束。

三步命令，在服务器项目根目录执行：

```bash
# 1. 固定一个AR检查点，为train/val全部窗口与七种mask生成缓存。
python -m unified_hoi.token_ar.prepare_refinement \
  --checkpoint runs/behave_10.27/checkpoint-0001000 \
  --output runs/refinement_cache/behave_step1000 \
  --device cuda:0 --dtype bfloat16 --temperature 0

# 中断后用相同参数加 --resume；--limit 仅供小窗口流程验证，不能当正式实验。
# 缓存目录完成后有COMPLETE标记；不改原AR、VQ或原生数据。

# 2. 训练小型修正网络；自动保存初始基线、best.pt、last.pt和逐次验证记录。
python -m unified_hoi.token_ar.train_refiner \
  --config configs/token_ar/refiner_behave.yaml

# 续训：配置schedule_steps保持不变，可增加max_steps，优化器/调度器/抽样状态恢复。
# python -m unified_hoi.token_ar.train_refiner --config configs/token_ar/refiner_behave.yaml \
#   --resume runs/token_ar/behave/refiner_step1000/last.pt

# 3. 同一批AR样本，对比修正前后的重建指标。
python -m unified_hoi.token_ar.benchmark \
  --checkpoint runs/behave_10.27/checkpoint-0001000 \
  --refiner runs/token_ar/behave/refiner_step1000/best.pt \
  --split val --temperature 0 --device cuda:0 --dtype bfloat16 \
  --output runs/eval_behave_step1000_refined
```

`summary.json`中`reconstruction`/`contact_micro`是修正后，`coarse_reconstruction`/`coarse_contact_micro`是同批次修正前；`generation`只报告修正后生成分布指标。没有`--refiner`时保持原路径。完整验证几何损失选最优；初始零残差模型也作为候选，防止只保存比初始更差的模型。默认5000步、warmup+cosine、8次验证无改善早停只是初始设置，未宣称收敛或优于基线。

`sample`同样支持`--refiner`，使用上面缓存时需加`--temperature 0 --text-conditioned`。采样温度/top-k（随机采样时）和文本开关必须匹配缓存设置。输出NPZ保留`coarse_human/object/contact`和修正后结果；保存的VQ token对应粗生成值，不能用于重新解码得到修正后运动。`evaluate`可直接复评修正结果并保留修正检查点SHA256。

修正检查点绑定原AR metadata、VQ、接触支持和身体资产哈希；1000步训练的修正器不能直接附加到24000步AR。缓存训练还验证原生样本和物体网格哈希。跨数据集不能混用；OMOMO配置为`configs/token_ar/refiner_omomo.yaml`，需要先用其AR检查点制作对应缓存。

验证：`python -B -m unittest discover -s tests -p 'test_refinement*.py' -v`。覆盖两种身体表示、七种mask条件保持、SO(3)合法性、几何梯度、冻结边界、检查点绑定、真实SMPL-H/SMPL-X资产、小型随机Qwen的缓存/训练/精确续训/采样/复评/配对benchmark。CPU流程验证不证明4B生成质量或正式训练后的收益。
