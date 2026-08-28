# A 股金融项目 3 天最小优化计划（秋招简历版）

> 修订日期：2026-08-28  
> 定位：只完成能够直接增强秋招简历与面试证据的必要工作，不再扩展技术栈。  
> 核心目标：形成“Risk 模型效果 → 接入 News Agent → 端到端验证”的完整证据链。  
> Sentiment、Single-Agent vs Multi-Agent 消融、30 条 Benchmark、复杂 LLM-as-Judge 均不再执行。

---

# 0. 是否可以只做第一件事？

可以。

如果时间非常紧，只完成 **Risk 模型实验**，已经能够得到当前项目最缺少、也是最有价值的一条量化证据：

> **Qwen3-8B + QLoRA 在固定 500 条测试集上的 Macro-F1 为 XX.X%，较 Base Model 提升 X.X pct。**

这已经足以支撑简历中的“金融新闻风险建模”部分。

但如果还能再投入约 1 天，建议继续完成第 2、3 项，因为这样项目能够形成：

```text
金融新闻
   ↓
Qwen3-8B QLoRA 风险分类
   ↓
News Agent
   ↓
其他投研 Agent
   ↓
Summary Agent
   ↓
最终投资分析报告
```

这样项目就不只是“训练了一个模型 + 做了一个 Agent”，而是一条完整系统链路。

---

# 1. 最终只做三件事

## 任务 1：完成 Risk 模型实验

### 目标

完成：

- TF-IDF + Logistic Regression
- Qwen3-8B Base
- Qwen3-8B + QLoRA

三种模型在 **同一固定 500 条 Risk 测试集**上的正式比较。

最终必须得到：

- Macro-F1
- Accuracy
- 平均推理延迟
- QLoRA 相对 Base 的提升

最重要的简历数字：

> **QLoRA 较 Base Model Macro-F1 提升 X.X pct。**

---

## 任务 2：把 Risk 模型接入 News Agent

### 目标

让 News Agent 真正使用已经训练好的 Risk QLoRA 模型，而不是只调用通用 LLM。

最终链路：

```text
新闻标题 + 新闻摘要
        ↓
Risk QLoRA
        ↓
风险等级 / 风险分数
        ↓
News Agent
        ↓
Summary Agent
```

---

## 任务 3：做一次端到端 Agent 验证

### 目标

不做复杂 Benchmark，也不做 Single-Agent / Multi-Agent 消融。

只准备少量代表性股票，确认整条系统链路可以正常运行：

```text
股票代码
   ↓
MCP 金融数据
   ↓
基本面 / 技术面 / 估值 / 新闻 Agent
   ↓
News Agent 使用 Risk 模型结果
   ↓
Summary Agent
   ↓
最终 Markdown 投资报告
```

验证系统“能够工作”即可，不再追求论文级系统评测。

---

# 2. 总工期

推荐：

> **3 天完成。**

如果只做任务 1：

> **约 2 天完成。**

建议优先级：

```text
Risk 模型实验
    ↓
Risk 接入 News Agent
    ↓
端到端验证
    ↓
停止优化
```

---

# 3. Day 1：冻结 Risk 数据与完成 LR / Base

## Step 1：只保留 Risk 数据

本轮不再做 Sentiment。

使用已经确定的 Risk 数据清洗规则：

- 使用 `Article_title`
- 使用 `Lsa_summary`
- 使用 `Stock_symbol`
- 使用 `Date`
- 标签仅接受 1～5
- 删除缺失关键字段样本
- 对完全重复新闻进行精确去重
- 使用时间切分 Train / Val / Test

如果现有正式 Risk split 已经生成并验收：

> 直接复用，不重新设计数据协议。

---

## Step 2：固定 500 条 Risk 测试集

从 Risk Test 中固定抽取：

```text
eval_test.csv
N = 500
random_state = 42
```

要求：

- 三个模型使用完全相同的 500 条样本；
- 样本冻结后不得重新抽取；
- 不允许根据结果修改测试集；
- Test 不参与训练和调参。

---

## Step 3：统一模型评测代码

统一输出：

```text
Macro-F1
Accuracy
Mean Latency
```

统一保存：

```text
sample_id
gold_label
prediction
latency
```

Qwen Base 和 QLoRA 使用相同：

- Prompt
- 输出解析器
- decoding 参数
- max input
- max output
- 500 条 eval_test

保证两者主要变量只有：

> 是否加载 QLoRA Adapter。

---

## Step 4：完成 TF-IDF + LR

训练：

```text
TF-IDF
   +
Logistic Regression
```

使用 Train 训练、Val 选择参数。

只进行最小参数比较：

```text
C ∈ {0.5, 1, 2}
```

选出 Val Macro-F1 最好的配置后冻结。

然后在固定 500 条 `eval_test.csv` 上得到：

- Macro-F1
- Accuracy
- Mean Latency

---

## Step 5：完成 Qwen3-8B Base

使用 Qwen3-8B Base 对固定 500 条测试集进行推理。

记录：

- Gold Label
- Prediction
- Raw Output
- Macro-F1
- Accuracy
- Mean Latency

完成后得到第一组核心对照：

```text
LR
vs
Qwen3-8B Base
```

---

# 4. Day 2：完成 QLoRA 与最终模型结果

## Step 1：确认 QLoRA Adapter 是否可正式使用

如果旧 Adapter 无法证明：

- 训练数据来源；
- Train / Test 无样本重叠；
- 训练代码版本；
- 训练数据切分方式；

则：

> 不使用旧 Adapter 作为正式结果，直接重新训练。

---

## Step 2：重新训练 Risk QLoRA

从正式 Risk Train 中固定采样约：

```text
10,000 ～ 20,000 条
```

原则：

- 固定随机种子；
- 尽量保持标签分布；
- Val 用于训练过程选择；
- Test / eval_test 不参与调参。

模型：

```text
Qwen3-8B
+
4-bit QLoRA
```

不做大范围超参数搜索。

---

## Step 3：正式评测 QLoRA

QLoRA 使用与 Base 完全相同的：

```text
500 条 eval_test
Prompt
Parser
Decoding
Metrics
```

输出：

- Macro-F1
- Accuracy
- Mean Latency

---

## Step 4：生成最终模型表

最终只保留这一张表：

| Model | Macro-F1 | Accuracy | Mean Latency |
|---|---:|---:|---:|
| TF-IDF + LR | XX | XX | XX |
| Qwen3-8B Base | XX | XX | XX |
| Qwen3-8B + QLoRA | **XX** | **XX** | XX |

计算：

```text
Macro-F1 绝对提升
=
F1_QLoRA - F1_Base
```

简历优先使用：

> **Macro-F1：XX.X% → XX.X%，QLoRA 较 Base 提升 X.X pct。**

这里优先报告 **百分点（pct）绝对提升**，不必再写复杂的相对百分比提升。

---

# 5. Day 2～3：Risk 模型接入 News Agent

完成模型实验后，再进行系统接入。

## Step 1：封装 Risk Predictor

将 QLoRA 推理封装为独立接口，例如：

```python
predict_risk(title, summary)
```

输入：

```text
title
summary
```

输出至少包含：

```text
risk_label
```

可选：

```text
raw_output
latency
```

---

## Step 2：修改 News Agent

News Agent 获取新闻后：

```text
MCP / 新闻数据
      ↓
提取新闻 title + summary
      ↓
Risk Predictor
      ↓
risk_label
      ↓
News Agent
```

News Agent 在生成新闻分析时明确加入：

- 新闻风险等级；
- 高风险新闻说明；
- 风险结果对整体判断的影响。

---

## Step 3：把 Risk 结果传递给 Summary Agent

保证最终状态中包含类似：

```text
news_analysis
risk_label
risk_summary
```

Summary Agent 生成最终报告时必须读取这些字段。

---

## Step 4：检查最终报告

最终报告中必须能够看到：

```text
新闻风险等级 / 风险判断
```

并且这个数字必须来自 Risk QLoRA，而不是 Summary Agent 自己重新猜测。

完成后形成：

```text
新闻
 ↓
QLoRA Risk
 ↓
News Agent
 ↓
Summary Agent
```

---

# 6. Day 3：端到端 Agent 验证

不建立 30 条 Benchmark。

只做：

> **5～10 个代表性股票案例。**

---

## Step 1：选择股票

选择约 5～10 只不同类型 A 股即可，例如覆盖：

- 银行
- 消费
- 新能源
- 半导体
- 医药

不需要追求行业完全覆盖。

---

## Step 2：逐只运行完整系统

每只股票执行完整流程：

```text
输入股票代码
     ↓
MCP 数据
     ↓
Fundamental Agent
Technical Agent
Value Agent
News Agent
     ↓
Summary Agent
     ↓
最终 Markdown 报告
```

---

## Step 3：检查四项内容

每个案例只检查：

### ① MCP 是否正常

确认金融数据工具可以正常返回真实数据。

### ② 四个分析 Agent 是否正常

确认：

- Fundamental
- Technical
- Value
- News

都能返回有效分析结果。

### ③ Risk 是否真正进入 News Agent

检查：

```text
Risk Predictor 输出
```

是否被 News Agent 使用。

### ④ Summary 是否使用 Risk 结果

最终报告中必须出现基于 Risk 模型得到的新闻风险判断。

---

## Step 4：保存少量案例

只保存：

```text
5～10 个完整结果
```

建议选择其中：

```text
2～3 个代表性案例
```

放进 README。

不计算：

- LLM-as-Judge
- Single-Agent vs Multi-Agent
- Agent Accuracy
- 复杂事实一致率
- 复杂 P95 Benchmark

只证明：

> 系统链路真实可运行。

---

# 7. 最终必须得到的结果

完成后只需要保留三类结果。

## 结果 1：模型实验

```text
TF-IDF + LR
Qwen3-8B Base
Qwen3-8B + QLoRA
```

指标：

```text
Macro-F1
Accuracy
Mean Latency
```

最重要数字：

> **QLoRA 较 Base Macro-F1 提升 X.X pct。**

---

## 结果 2：系统架构

能够明确展示：

```text
MCP 金融数据
       +
Risk QLoRA
       ↓
4 个投研 Agent
       ↓
Summary Agent
       ↓
投资报告
```

---

## 结果 3：端到端案例

保存 5～10 个成功运行案例，证明：

- MCP 数据正常；
- 四个 Agent 正常；
- Risk 模型正常；
- News Agent 使用 Risk；
- Summary Agent 输出最终报告。

---

# 8. 明确不做

本轮全部删除：

- Sentiment 模型实验
- Sentiment QLoRA
- Single-Agent
- Single-Agent vs 5-Agent 消融
- 30 条 Agent Benchmark
- Fixture Backend
- Agent Benchmark 评分器
- LLM-as-Judge
- Agent 完整率
- Agent 事实一致率
- Agent 工具调用成功率 Benchmark
- Agent P50 / P95 正式压测
- 大规模稳定性测试
- Embedding / MinHash / SimHash 近似新闻去重
- 200～500 条人工 Gold Dataset
- Bootstrap
- 显著性检验
- QWK
- 大范围 QLoRA 调参
- 新增 Agent
- 新增模型
- 新增 RAG
- 新增 Memory

---

# 9. 最终停止条件

满足以下条件立即停止 Finance 优化：

- [ ] Risk 固定 500 条 eval_test
- [ ] TF-IDF + LR 完成
- [ ] Qwen3-8B Base 完成
- [ ] Risk QLoRA 完成
- [ ] Macro-F1 / Accuracy / Mean Latency 完成
- [ ] 得到“QLoRA 较 Base Macro-F1 提升 X.X pct”
- [ ] Risk Predictor 接入 News Agent
- [ ] Risk 结果进入 Summary Agent
- [ ] 完成 5～10 个端到端案例
- [ ] README 更新
- [ ] 简历更新

完成后：

> **停止增加任何新技术模块。**

---

# 10. 时间不足时的最小停止线

如果只能继续优化约 2 天，则只完成：

```text
Risk 数据 / 500 条测试集冻结
           ↓
TF-IDF + LR
           ↓
Qwen3-8B Base
           ↓
Qwen3-8B QLoRA
           ↓
Macro-F1 / Accuracy / Latency
           ↓
简历数字
```

然后停止。

这仍然是一个完整、可信、可以用于秋招简历的模型实验。

此时简历重点写：

> 基于金融新闻构建 1～5 级风险分类任务，对 Qwen3-8B 开展 4-bit QLoRA 微调；在固定 500 条测试集上 Macro-F1 达 XX.X%，较 Base Model 提升 X.X pct，并与 TF-IDF+LR 基线进行对比。

不再强调 Agent 的量化提升，只把现有 Multi-Agent 系统作为项目工程实现进行描述。
