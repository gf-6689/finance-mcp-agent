# 模型对比（固定 500 条 Risk eval_test）

> 评测口径：同一冻结 `data/splits/risk/eval_test.csv`（N=500，SHA256
> `b052920047da97cc3e7e7ab4df382e79ebc9ab49e4fd466e7364807972272986`）。
> Macro-F1 固定 labels=[1,2,3,4,5]、average="macro"、zero_division=0。

| Model | Macro-F1 | Accuracy | Mean Latency |
| ---------------- | -------: | -------: | -----------: |
| TF-IDF + LR      | 0.340475 | 0.584000 |  0.197733 ms |
| Qwen3-8B Base    | 0.272622 | 0.338000 |  133.973080 ms |
| Qwen3-8B + QLoRA | 未运行 | 未运行 | 未运行 |

## 说明

- TF-IDF + LR：selected_C=2.0（仅按 Val Macro-F1 选择）；延迟计时范围 =
  TF-IDF transform + LR predict。
- Qwen3-8B Base：本地 Qwen3-8B 基座（与未来 QLoRA 相同的 checkpoint），
  `enable_thinking=False`、`do_sample=False`、`max_new_tokens=32`；
  延迟计时范围 = `model.generate`（torch.cuda.synchronize 前后包裹），
  不含模型/CSV 加载。
- QLoRA 行在本阶段不填写任何数字。
