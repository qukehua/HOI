# 当前模型保留的上游代码

`src/unified_hoi/integrations/_tridi_primitives.py` 保留 TriDi 的 `Projection` 与 `get_timestep_embedding` 原函数体，当前参数模型直接导入它们。完整 MIT 许可证位于 `src/unified_hoi/integrations/licenses/TRIDI_LICENSE.txt`，必须随代码保留。

来源为 [ptrvilya/tridi](https://github.com/ptrvilya/tridi)，固定提交 `afa9631dc2b3a250588ab64026eeaa37f18f0d38`，文件 `tridi/model/denoising/transformer_uni_3.py`，Git blob 为 `9115e4abe94956d0f6b8708ca6d1a734fbd4d2ad`。本地副本的来源信息同时写在代码头部。

BEHAVE 的690个表面点索引已保存于 `data/parametric_assets/behave/contact_support.npz`；OMOMO 使用独立的SMPL-X点集。训练、采样、评估只读取这些已准备资产。按用户要求，`external/` 下的 Kimodo、TriDi、OMOMO 参考仓库及全部内容保留。项目源码中的旧 Kimodo 适配器仍列入代码清理清单。

接触AE与HOI模型是本项目独立训练的权重，不与上游检查点互换。源代码许可证不替代人体资产、数据集或模型权重的使用条款。当前模型和评估限制见 [原生参数重训说明](PARAMETRIC_RETRAINING.md)。
