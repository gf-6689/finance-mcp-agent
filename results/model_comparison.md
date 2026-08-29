# 模型对比（固定 500 条 Risk eval_test）

> 评测口径：同一冻结 `data/splits/risk/eval_test.csv`（N=500，SHA256
> `b052920047da97cc3e7e7ab4df382e79ebc9ab49e4fd466e7364807972272986`）。
> Macro-F1 固定 labels=[1,2,3,4,5]、average="macro"、zero_division=0。

| Model | Macro-F1 | Accuracy | Mean Latency |
| ---------------- | -------: | -------: | -----------: |
| TF-IDF + LR      | 0.340475 | 0.584000 |  0.197733 ms |
| Qwen3-8B Base    | 0.272622 | 0.338000 |  133.973080 ms |
| Qwen3-8B + QLoRA | 0.513218 | 0.730000 |  220.646619 ms |

## 关键结论

- QLoRA 较 Base 的 Macro-F1 绝对提升：
  `delta = 0.5132181966 - 0.2726223599 = 0.2405958367`
  → **提升 24.06 个百分点**。
- QLoRA Macro-F1（0.513218）超过 TF-IDF + LR 基线（0.340475），
  同时 Accuracy（0.730）也高于 LR（0.584）。
- QLoRA 延迟高于 Base：220.65 ms vs 133.97 ms（同为 BF16 逐样本
  `model.generate` 口径；QLoRA 为 base + LoRA 双路前向，未经 merge）。
- 冻结 eval_test 中 label 5 仅有 6 条，且 QLoRA 对 label 5 的预测数为 0；
  因此不能声称五个风险等级都获得了稳定改善。

## 说明

- TF-IDF + LR：selected_C=2.0（仅按 Val Macro-F1 选择）；延迟计时范围 =
  TF-IDF transform + LR predict。
- Qwen3-8B Base：本地 Qwen3-8B 基座（与 QLoRA 相同的 checkpoint），
  `enable_thinking=False`、`do_sample=False`、`max_new_tokens=32`；
  延迟计时范围 = `model.generate`（torch.cuda.synchronize 前后包裹），
  不含模型/CSV 加载。
- Qwen3-8B + QLoRA：正式训练 20,000 条 × 3 epochs（4-bit QLoRA，
  nf4 + 双重量化，LoRA r=16/alpha=32，seed=42）；三个 epoch checkpoint
  由冻结 Val1000 生成式评测选出（epoch 2 / global_step 2500，
  Val Macro-F1 = 0.502214，规则 = 仅按 Val Macro-F1，并列取更小
  global_step）；正式评测为 BF16 base + 选中 Adapter，Prompt/Parser/
  decoding/计时口径与 Base 完全一致（valid_output_rate = 1.000，
  invalid = 0）。确定性复核（独立目录、相同配置）结果逐位一致。
