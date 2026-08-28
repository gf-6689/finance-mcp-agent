# A 股金融项目 6 天优化执行基线（简历证据链版 v2）

> 建立日期：2026-08-24  
> 修订日期：2026-08-28
> 状态：**唯一执行基线（后续所有优化工作以本文档为准）**  
> 定位：目标是把项目补到“秋招简历可写、面试可解释”的程度，**不是**论文级评测。  
> 默认版本：6 天。第 7 天仅作为 QLoRA 重训、失败重跑与收尾缓冲，**不扩大实验范围**。
>
> **Day 1 当前状态：规则已冻结，核心逻辑已通过测试；正式 CSV、manifest、report、spotcheck 生成并验收后，才标记为完成。**

---

## 0. 结论先行：两条证据链，两张结果表

整个优化只回答两个问题，其余工作都围绕它们服务：

| 证据链 | 回答的面试问题 | 最终产出 |
|---|---|---|
| ① TF-IDF+LR / Qwen Base / QLoRA 同评测集对比 | QLoRA 微调到底有没有用？ | 表 1（见 §5） |
| ② Single-Agent / 5-Agent 同输入消融 | 为什么要 Multi-Agent？ | 表 2（见 §5） |

**最低成功线：Day 1 全部产物生成并验收 + Risk 的 TF-IDF+LR / Qwen3-8B Base / QLoRA 同协议对比 + 30 条 Agent Benchmark + `final_numbers.md` / README / 简历更新。Sentiment 不进入最低成功线。**

---

## 1. 优化目标与明确不做的内容

### 1.1 优化目标

1. 建立**可信、无完全重复样本泄漏、固定**的 Train / Val / Test 划分。
2. 从 Test 中固定抽取一份 `eval_test.csv`，供 LR / Base / QLoRA 使用完全相同的评测样本。
3. 得到模型对比表：LR vs Qwen3-8B Base vs QLoRA，Risk 为 P0；Sentiment 只在其他 P0 全部完成后有余力时执行。主指标仅保留 Macro-F1 + MAE，附 Accuracy 与 Confusion Matrix。
4. 得到 Agent 系统级架构消融表：Single-Agent vs 5-Agent，使用 30 条固定问题、相同工具接口、相同 fixtures 数据；指标仅保留完整率、事实一致率、工具成功率、P50/P95 延迟。
5. 产出可投递 README、3 张图和可直接写入简历的真实数字。

### 1.2 明确不做的内容（P1，默认不执行）

以下条目**不属于**本次优化范围，除非两条证据链全部完成且仍有剩余时间：

- 相似新闻 embedding / MinHash / SimHash 聚类去重（本轮只做精确去重）
- 200～500 条人工标注、双人标注、Cohen's Kappa
- QWK、Bootstrap 置信区间、显著性检验、RMSE 等扩展指标
- 60～100 条 Agent 问题
- 大规模重复稳定性实验（Sentiment 之后仍有余力，最多 5～10 条综合问题 × 3 次）
- 自动通用事实核验（只核验预先定义的固定数字字段）
- Token Cost 统计
- 大范围 QLoRA 超参数搜索
- 金融数据版本管理系统（用 fixtures JSON + manifest 替代）
- 新增模型、新增 Agent、新增工具、新增技术模块

### 1.3 已有简历数字的处理原则

| 简历候选数字或表述 | 本轮处理 |
|---|---|
| Risk / Sentiment Accuracy `91% / 88%` | 暂停使用；旧随机切分结果不作为正式简历数字，等待同协议重新验证 |
| 工具调用成功率 `98%` | 有逐调用日志和明确分母则复核；否则以新 Benchmark 结果替换，并注明固定 30 题与 fixture 范围 |
| 数据一致率约 `99%` | 有 ground truth / fixture 证据则复核；否则以新 Benchmark 的事实一致率替换 |
| `2 分钟+ → 约 90 秒`、效率提升约 `1.25 倍` | 仅在存在同场景、同模型、同环境的前后原始日志时保留；否则只报告新 Benchmark 的 P50/P95 |
| 覆盖 A 股 `5000+` 股票 | 仅作为经数据源或工具股票列表验证后的能力范围，不与 30 题 Benchmark 指标混写 |
| 用户满意度接近 `90%` | 仅在确有真实灰度记录时使用；本仓库不补造 |
| `500+` 高频问题知识库 | 仅在确有数据文件或业务记录时使用；本轮不新增 |
| SFT + GRPO、奖励提升 `3–5%` | 当前个人仓库不作为证据；无独立真实记录时不写 |
| “整理并标注约 10 万条” | 改为“清洗并构建约 10 万条金融新闻弱标注数据”；只有存在真实人工标注流程与记录时才写“人工标注” |

所有百分比最终以可追溯的新数字替换，不以复现旧简历数字为优化目标。

---

## 2. 当前项目存在的问题

以下问题为本计划的出发点：

1. **数据划分不可信**：当前训练脚本使用随机切分，且缺少严格去重与时间切分，近重复新闻可能跨集合。
2. **无固定测试集**：实验间缺少统一、冻结的测试协议。
3. **无统一评测脚本与统一输出解析**：Base / QLoRA 输出格式可能不同，指标口径存在不一致风险。
4. **无传统 ML 基线**：缺少 TF-IDF + LR，无法回答“大模型是否真的必要”。
5. **无 Base vs QLoRA 正式对照**：现有 Adapter 未在同一 Prompt、同一评测集、同一解析器下做正式比较。
6. **旧 Adapter 缺少完整可追溯训练清单**：默认重训。只有旧数据文件哈希、训练代码版本、切分 seed 与旧训练 `sample_id` 均可重建时，才允许进入复用检查。
7. **Agent 无固定 Benchmark**：Multi-Agent 效果目前没有可复现的定量证据。
8. **MCP 实时数据不可直接比较**：Single 与 Multi 若拿到不同实时结果，架构消融不成立。
9. **Agent 完整率缺少逐题定义**：不同问题需要的分析模块不同，不能统一要求所有任务包含六个模块。
10. **失败任务若从分母删除会高估结果**：正式 Benchmark 冻结后，崩溃、超时、异常必须保留并记为失败。
11. **数据域表述风险**：新闻模型使用**纳斯达克英文新闻**，Agent 面向 **A 股**；README 与简历必须明确区分。

---

## 3. 总体执行纪律

1. **测试协议冻结**  
   Day 1 生成 Train / Val / Test 后，通过 `sample_id` 清单、文件 SHA256、样本数和日期范围写入 `split_manifest.json`。大型 CSV 本身不要求提交 Git。

2. **统一评测集**  
   从 Test 中固定分层抽取 `eval_test.csv`，三种模型必须使用完全相同的样本。

3. **禁止用 Test 调参**  
   LR 的 `C`、QLoRA 训练设置、Prompt 调整只能依据 Train / Val。Test 和 `eval_test` 只用于最终比较。

4. **控制变量**  
   - Base 与 QLoRA：同 Prompt、同 parser、同 `eval_test`、同推理参数。
   - Single 与 Multi：同 LLM、同问题、同工具接口、同 fixtures、同总超时、同重试规则、同输出上限，并记录 LLM 与工具调用预算。

5. **如实报告**  
   不伪造、不挑样本、不删除失败任务。若结果不支持原假设，同样保留。

6. **范围冻结**  
   执行期间不新增技术模块。任何新增工作必须直接服务两条证据链，否则归入 P1。

7. **结论边界**  
   本轮只能声称“无完全重复样本泄漏，并通过时间切分降低事件泄漏风险”。未做事件级近似去重时，不得声称绝对“无数据泄漏”。

8. **评分协议前置冻结**
   Agent 正式运行前必须冻结 30 道题、fixtures、`required_sections`、`required_facts`、完整率判定规则、事实容差及失败处理。正式输出生成后不得修改题目、样本、评分规则或事实容差。

9. **弱标注口径**
   `risk_deepseek` / `sentiment_deepseek` 数据统一称为“清洗并构建的金融新闻弱标注数据”。没有真实人工标注流程和记录时，不得写成“人工标注约 10 万条”。

---

# 4. Day 1～Day 6 每日任务

## Day 1：数据重构 + 测试协议冻结（P0，最不能省）

### 任务

#### 1. 明确标签语义

首先在 README / 配置中固定：

- `risk_deepseek`：1～5 的具体风险等级含义
- `sentiment_deepseek`：1～5 的具体情感等级含义

训练、Prompt、人工抽查、评测必须使用同一含义。

#### 2. 写 `preprocess/make_splits.py` 并冻结数据协议

##### 2.1 标准字段契约

源字段映射固定为：

```text
title   = Article_title
summary = Lsa_summary
symbol  = Stock_symbol
url     = Url
date    = Date
```

正式输出字段固定为：

```text
sample_id,date,title,summary,stock_symbol,url,label
```

规则：

- `Article_title`、`Lsa_summary`、`Stock_symbol`、`Date` 缺失或清洗后为空时删除
- `Url` 可缺失，不因 URL 缺失删除
- 不使用 `Article` 正文、其他摘要列或其他摘要算法回退
- Risk / Sentiment 标签只接受可无损转换为整数的 1～5
- Sentiment 的缺失标签和非法 `0.0` 删除
- `Publisher`、`Author`、`Article` 和其他摘要列不进入正式 split

##### 2.2 标准化

- 文本：NFKC → `casefold()` → 连续空白压成单空格 → `strip()`
- `stock_symbol`：NFKC → `strip()` → `upper()`
- URL：NFKC → `strip()`
- URL 不修改 path 大小写和 query 参数

##### 2.3 精确去重与冲突标签

去重粒度固定为 `stock_symbol` 任务粒度，阶段顺序固定为：

```text
URL → title+summary → title
```

每个阶段均执行：

```text
groupby(stock_symbol + key)
→ 检查 label
→ 标签一致：保留日期最早记录
→ 标签冲突：整组删除
```

补充规则：

- 空 URL 跳过 URL 阶段，但继续进入后续去重阶段
- 排序固定为 `date ASC → original_row_id ASC`
- 使用稳定排序

##### 2.4 `sample_id`

定义固定为：

```text
sha256(
    normalized_stock_symbol + "\n" +
    normalized_title + "\n" +
    normalized_summary
)
```

写出前必须断言：

- `sample_id` 无缺失
- 每个任务文件内 `sample_id` 唯一

报告不得声称“全局事件级无泄漏”，只能表述为：

> 在股票任务粒度进行精确去重，并通过时间切分降低事件泄漏风险。

#### 3. 按日期边界切分，而不是按行号硬切

流程：

```text
按日期排序
    ↓
寻找接近 70% 的日期 cutoff_1
    ↓
寻找接近 85% 的日期 cutoff_2
    ↓
Train: date <= cutoff_1
Val:   cutoff_1 < date <= cutoff_2
Test:  date > cutoff_2
```

原则：

- 同一天的数据不跨集合。
- Test 表示未来时间段。
- 相同 `sample_id` 不得跨集合。

#### 4. 从 Test 固定抽取 `eval_test.csv`

每个任务的正式规模固定为 **500 条**，冻结后不再扩容或缩减。规则：

- 标签集合固定为 `[1,2,3,4,5]`
- 每个 Test 中存在类别的最低目标为 10 条
- 类别不足 10 条时全部保留
- Risk label 5 当前 Test 仅 6 条，因此 6 条全部进入
- 不复制、不放回
- 先按 Test 自然标签比例计算 ideal quota
- 使用动态差值分配或扣减，每调整 1 个名额后重新计算
- 若 `test_total < 500` 立即报错
- 固定 `random_state=42`
- 各类别完成抽样后合并，并按 `sample_id` 升序写出
- 一次冻结后不得根据模型结果重新抽样

`split_report.md` 必须同时记录 Test 与 `eval_test` 的标签分布及抽样规则。Macro-F1 固定按标签集合 `[1,2,3,4,5]` 计算，缺失类别 F1 记 0，`zero_division=0`。

> `eval_test.csv` 只服务正式模型对比，不用于任何调参。

#### 5. 生成冻结清单

`split_manifest.json` 结构固定采用：

```text
顶层 protocol
+ generator
+ tasks.risk
+ tasks.sentiment
```

`generator` 至少包含：

```json
{
  "script": "preprocess/make_splits.py",
  "script_sha256": "",
  "git_commit": ""
}
```

要求：

- `script_sha256` 必填
- `source_sha256` 必填
- Risk / Sentiment 各自记录清洗、删除、冲突和标签分布统计
- 每个 train / val / test / eval_test 同时记录 CSV SHA256 和 `sample_ids_sha256`
- `sample_ids_sha256` 对排序后的唯一 `sample_id` 集合计算
- 每个 split 同时记录样本数和日期范围
- `split_report.md` 记录执行时 Git working tree 是否 dirty
- 不记录 manifest 自身哈希

大型 CSV 可 `.gitignore`，Git 提交：

- split 脚本
- manifest
- `sample_id` 清单或哈希
- 统计报告

#### 6. 标签分层抽查

总量控制在约 100 条：

- Risk：1～5 每档约 10 条
- Sentiment：1～5 每档约 10 条

只判断是否存在明显大面积误标，不建立人工 Gold Dataset。

### 预期文件

| 文件 | 说明 |
|---|---|
| `preprocess/make_splits.py` | 清洗、精确去重、sample_id、时间切分 |
| `data/splits/risk/{train,val,test,eval_test}.csv` | 风险划分 |
| `data/splits/sentiment/{train,val,test,eval_test}.csv` | 情感划分 |
| `data/splits/split_manifest.json` | 测试协议冻结信息 |
| `data/splits/split_report.md` | 去重数、日期范围、标签分布 |
| `data/splits/label_spotcheck.md` | 分层抽查结论 |

### 验收标准

- [ ] 标签 1～5 语义已固定
- [ ] Train / Val / Test 不存在重复 `sample_id`
- [ ] Test 按时间位于 Train / Val 之后
- [ ] `eval_test.csv` 已固定
- [ ] 正式 Train / Val / Test / eval_test CSV 已生成并验收
- [ ] `split_manifest.json` 已提交 Git
- [ ] `split_report.md` 与 `label_spotcheck.md` 已生成并验收
- [ ] 分层抽查未发现该标签维度大面积不可用

### 不做

MinHash、Sentence-BERT、事件级聚类等近似去重全部放入 P1。

---

## Day 2：统一评测器 + TF-IDF + LR（P0）

### 任务

#### 1. 写 `eval/metrics.py`

统一实现：

- Macro-F1
- MAE
- Accuracy
- Confusion Matrix

后续 LR / Base / QLoRA 全部复用。

#### 2. 写统一输出解析器 `eval/parser.py`

Qwen Prompt 统一要求最终输出：

```text
FINAL_LABEL=4
```

解析器只接受：

```text
FINAL_LABEL=[1-5]
```

无法解析时记为：

```text
invalid_prediction
```

规则：

- invalid 不允许静默删除
- 记作错误预测
- 额外记录 `valid_output_rate`

`valid_output_rate` 只作为诊断指标，不要求进入简历主表。

#### 3. 训练 TF-IDF + LR

推荐固定：

```python
TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=50000
)

LogisticRegression(
    class_weight="balanced"
)
```

Validation 只比较：

```text
C ∈ {0.5, 1, 2}
```

选择最优 C 后冻结。Risk 必须完成；Sentiment LR 成本较低，可以生成，但不得阻塞 Risk 三模型对比和 Agent Benchmark。Sentiment Base / QLoRA 只在其他 P0 全部完成后执行。

#### 4. 正式比较时使用相同 `eval_test.csv`

LR 虽然可以轻松跑完整 Test，但**表 1 中三种模型必须使用相同的 `eval_test.csv`**，避免样本规模不同造成不可比。

完整 Test 上的 LR 结果可以保留为附加结果，但不进入主表。

### 预期文件

| 文件 | 说明 |
|---|---|
| `eval/metrics.py` | 统一指标 |
| `eval/parser.py` | 统一 1～5 输出解析 |
| `eval/baseline_tfidf_lr.py` | LR 基线 |
| `results/tfidf_lr/{risk,sentiment}_metrics.json` | 正式指标 |
| `results/tfidf_lr/{risk,sentiment}_confusion.png` | 混淆矩阵 |

### 验收标准

- [ ] Risk 的 Macro-F1、MAE、Accuracy 已生成
- [ ] Sentiment LR 可选；Sentiment Base / QLoRA 仅在其他 P0 全部完成后执行，未完成则在表 1 标为未运行
- [ ] 三模型共用同一指标代码
- [ ] 表 1 中 LR 使用固定 `eval_test`

---

## Day 3～4：Qwen3-8B Base + QLoRA（P0）

### 任务

#### 1. 写 `eval/prompt.py`

固定：

- 输入字段：股票代码（若数据存在）+ 新闻标题 + 新闻摘要
- 标签定义：与 Day 1 完全一致
- 输出格式：`FINAL_LABEL=X`
- `do_sample=False`

若推理框架在 `do_sample=False` 时忽略 temperature，则不要依赖 temperature 作为控制变量；以 greedy decoding 为准。

#### 2. 写 `eval/qwen_base.py`

加载 Qwen3-8B Base：

- 使用 `eval_test.csv`
- 使用统一 Prompt
- 使用统一 parser
- 保存逐样本预测、raw output、parsed label、耗时

#### 3. 旧 Adapter 默认重训与例外复用规则

当前旧训练流程未随 Adapter 保存完整训练 `sample_id` 清单和输入 CSV 哈希，因此正式实验默认从新 Train 重训。

只有同时具备以下证据时，才允许尝试复用旧 Adapter：

- 旧训练输入 CSV 的 SHA256
- 旧训练代码版本或 Git commit
- 数据读取顺序、过滤规则与切分 seed
- 可重建的旧训练 `sample_id` 清单

证据完整后仍须检查：

```text
old_adapter_train_sample_ids ∩ new_test_sample_ids
old_adapter_train_sample_ids ∩ new_eval_test_sample_ids
```

两个交集均为 0 才可复用。任一证据缺失、任一交集非 0，或无法证明训练时数据文件与当前文件一致，旧 Adapter 只能用于调试，不得进入正式表 1 或简历数字。

#### 4. 重训

从新的 Train 中固定采样：

- 推荐 1～2 万条
- 尽量保持标签分布
- 采样 seed 固定

复用原有 QLoRA 脚本，但改为读取新 Train / Val。

禁止：

- 依据 Test / eval_test 结果调整超参数
- 为了让数字更好反复更换训练子集

#### 5. 写 `eval/qwen_qlora.py`

Base 与 QLoRA 唯一主要变量：

```text
是否加载 Adapter
```

其余保持一致：

- eval_test
- Prompt
- parser
- decoding
- max input
- max output
- 量化设置
- 指标

### 预期文件

| 文件 | 说明 |
|---|---|
| `eval/prompt.py` | 统一 Prompt |
| `eval/qwen_base.py` | Base 推理 |
| `eval/qwen_qlora.py` | QLoRA 推理 |
| `results/qwen/base/...` | Base 结果 |
| `results/qwen/qlora/...` | QLoRA 结果 |
| `results/adapter_leak_check.md` | Adapter 泄漏检查记录 |
| `results/model_comparison.md` | 表 1 |

如需重训，再增加：

| 文件 | 说明 |
|---|---|
| `preprocess/sample_train.py` | 训练集固定采样 |
| `train/retrain_qlora_risk.py` | 风险 QLoRA 重训 |
| `train/retrain_qlora_sentiment.py` | 情感 QLoRA 重训 |

### 验收标准

- [ ] Base 与 QLoRA 使用相同 eval_test / Prompt / parser
- [ ] 默认重训；若复用旧 Adapter，则数据哈希、代码版本、切分规则和训练 `sample_id` 证据完整
- [ ] 证据缺失或任一交集非 0 时，旧 Adapter 未进入正式表 1
- [ ] 表 1 正式数字全部可追溯
- [ ] 相对提升计算完成：

\[
\frac{F1_{QLoRA}-F1_{Base}}{F1_{Base}}\times 100\%
\]

---

## Day 4：Agent Benchmark + Fixture Tool Backend（P0）

> QLoRA 重训可与本阶段并行，但不得挤占 5 条 smoke test 和 fixture 完整性验证。

### 任务

#### 1. 建立固定 Benchmark

默认目标固定为 **30 条 P0**，本轮不扩到 40 条。正式冻结前先使用 5 条覆盖不同任务类型的问题做 smoke test，验证 fixture 完整性、评分器和两套 runner；smoke 结果不进入正式表。

30 条版本：

| 类型 | 数量 |
|---|---:|
| 基本面 | 6 |
| 技术面 | 6 |
| 估值 | 6 |
| 新闻风险 | 6 |
| 综合投研 | 6 |
| 总计 | 30 |


股票覆盖：

- 6～10 只 A 股
- 至少覆盖银行、消费、新能源、半导体、医药、周期、科技等多个行业

正式冻结前必须完成人工审查：

- 两种架构都能访问完成任务所需的全部工具
- 题目不按现有 Multi-Agent 报告模板反向设计
- 每个事实字段在 fixture 中真实存在，并已固定字段路径、日期、单位和容差
- 完整率评分不依赖固定标题文字
- smoke 题不进入正式 30 题
- 评分规则在正式运行前冻结，正式输出生成后不得后验调题或调评分器

#### 2. 每题明确评分要求

`questions.jsonl` 每条必须包含：

```json
{
  "id": "q001",
  "stock_code": "600519",
  "query": "分析该公司当前估值水平",
  "task_type": "valuation",
  "required_sections": [
    "valuation",
    "conclusion"
  ],
  "required_facts": [
    "pe",
    "pb"
  ]
}
```

综合问题可以要求：

```json
{
  "required_sections": [
    "fundamental",
    "technical",
    "valuation",
    "news",
    "risk",
    "conclusion"
  ]
}
```

这样完整率按任务实际要求计算，不统一要求所有题包含六模块。

#### 3. 冻结外部数据

首次获取 Benchmark 所需 MCP 数据并保存为 fixtures。

**重要：Agent 不直接绕过工具读取 JSON。**

改为：

```text
Agent
  ↓
原有 Tool 接口
  ↓
Fixture Backend
  ↓
冻结 JSON
```

即工具函数的调用方式、名称、参数约束保持不变，只替换底层数据来源。

#### 4. 冻结所有实际依赖的数据

不仅限于：

- financial
- valuation
- news
- quote

还要覆盖 Benchmark 中技术分析等模块真实依赖的 K 线、指标或其他数据。

原则：

> Single 与 Multi 在每一道题上看到完全相同的外部数据。

### 预期文件

| 文件 | 说明 |
|---|---|
| `Financial-MCP-Agent/benchmark/questions.jsonl` | 30 条正式问题 |
| `Financial-MCP-Agent/benchmark/fixtures/*.json` | 冻结数据 |
| `Financial-MCP-Agent/benchmark/freeze_fixtures.py` | fixture 生成 |
| `Financial-MCP-Agent/benchmark/fixture_backend.py` | Tool → Fixture 数据后端 |

### 验收标准

- [ ] 5 条 smoke test 已通过，且未进入正式结果
- [ ] 30 条 P0 Benchmark 已冻结
- [ ] 每题都有 `required_sections`
- [ ] 需要数字核验的题有 `required_facts`
- [ ] Single / Multi 使用相同 Tool 接口和相同 fixtures
- [ ] 正式运行不回退实时 MCP 数据
- [ ] 题目、fixtures、评分规则和事实容差已在正式运行前冻结
- [ ] 正式输出生成后未修改题目、样本、评分规则或事实容差

---

## Day 5：Single-Agent vs 5-Agent 正式消融（P0）

> 当日必须完成 30 题两套架构的正式运行、失败日志和表 2 原始指标；不得把主要正式运行推迟到 Day 6。

### 任务

#### 1. 实现 Single-Agent

要求：

- 单一 Agent
- 与 Multi 使用同一个 LLM
- 能访问相同工具集合
- 使用相同 Fixture Backend
- 不调用现有分析子 Agent
- 自己完成工具选择、分析与最终报告

#### 2. Multi-Agent

调用现有 5-Agent LangGraph 流程：

- 基本面
- 技术面
- 估值
- 新闻
- 汇总

同样使用 Fixture Backend。

#### 3. 统一运行条件与结论边界

固定：

- LLM 与模型版本
- Benchmark 与 fixtures
- 可见工具集合及参数 schema
- 每题端到端总超时
- 最大重试次数
- 最大输出 token
- 运行环境与并发设置
- 输出结构要求

两套系统必须记录每题的 LLM 调用次数、工具调用次数、输入输出 token（若接口可得）和最终是否触发预算上限。优先设置相同的每题工具调用上限；若因现有 5-Agent 编排无法严格统一 LLM 调用次数，则保留真实差异并在结果中报告。

本实验比较的是完整系统架构，不是严格隔离 Prompt、调用次数和上下文组织方式后的纯 Agent 数量实验。正式结论统一表述为“系统级架构消融”。

#### 4. 指标只保留 4 类

### ① 报告完整率

对第 i 题：

\[
Complete_i=
\begin{cases}
1,& \text{required\_sections 全部覆盖}\\
0,& \text{否则}
\end{cases}
\]

整体：

\[
CompletenessRate=
\frac{\sum_i Complete_i}{N}
\]

判定规则必须在正式 Benchmark 冻结前写入评分配置：每个 section 配置允许的标题、同义表达及最低有效内容长度；不能仅以关键词出现判定覆盖。

注意：

- 任务超时 = 0
- Agent 崩溃 = 0
- 无有效报告 = 0
- 只有标题、占位文本或无实质内容 = 0
- **不得从分母删除**

### ② 事实一致率

只验证 `required_facts` 中预先规定的固定字段，例如：

- PE
- PB
- 营收
- 净利润
- 收盘价
- ROE

以 fixture 数据作为 ground truth。正式冻结前必须为每个事实定义：字段路径、财报或行情日期、展示单位、允许误差、百分比与小数换算规则，以及万/亿等单位换算规则。

判定规则：

- 仅含 `required_facts` 的题进入事实一致率分母
- 分母为所有预定义事实项数量，不是成功回答的事实数量
- 缺失、无法解析、单位错误、引用错误期间或同一报告内出现冲突值均记失败
- 超时、崩溃和无有效报告时，该题全部预定义事实项记失败

不开发通用事实核验器。

### ③ 工具调用成功率

\[
ToolSuccessRate=
\frac{成功返回的工具调用数}
{全部工具调用数}
\]

超时、非法参数、异常均记失败。

### ④ 延迟

记录：

- P50
- P95

使用端到端 wall-clock latency。正式表同时标注 `N=30`；P95 只作为该固定小样本 Benchmark 的描述性结果，不作稳定尾延迟结论。

### 预期文件

| 文件 | 说明 |
|---|---|
| `Financial-MCP-Agent/benchmark/run_single.py` | Single-Agent |
| `Financial-MCP-Agent/benchmark/run_multi.py` | Multi-Agent |
| `Financial-MCP-Agent/benchmark/metrics.py` | 统一指标 |
| `results/agent_ablation/single/*.json` | 逐题结果 |
| `results/agent_ablation/multi/*.json` | 逐题结果 |
| `results/agent_ablation/comparison.md` | 表 2 |

### 验收标准

- [ ] 所有正式 Benchmark 任务均保留在分母
- [ ] Single / Multi 的模型、数据、工具接口和运行约束一致
- [ ] LLM 与工具调用次数、预算上限均已记录
- [ ] 结果明确标注为系统级架构消融
- [ ] 表 2 全部填满
- [ ] 所有失败、超时、异常都有日志

### 不做

大规模重复运行稳定性实验。

若 Sentiment 之后仍有余力：

```text
5～10 条综合问题 × 3 次
```

仅作为附加验证，不影响 P0 完成判断。

---

## Day 6：成果整理与可信收尾（P0）

> Day 6 原则上不再新增实验，只整理已冻结结果、生成图表、核对数字并更新 README。仅允许重跑因基础设施故障且已有明确日志的任务，不得因结果不理想重抽样或改题。

### 任务

#### 1. README 只保留 5 个核心部分

```text
1. 项目介绍
2. 系统架构
3. 金融新闻风险 / 情感建模
4. Model Benchmark
5. Agent Ablation
```

必须明确：

- 新闻模型数据域：纳斯达克英文新闻
- Agent 数据域：A 股

不得混写成“使用 A 股新闻训练金融模型”。

#### 2. 只做 3 张图

### 图 1：模型 Macro-F1

```text
TF-IDF + LR
Qwen3-8B Base
Qwen3-8B + QLoRA
```

### 图 2：一张代表性 Confusion Matrix

优先选择最重要的风险任务；若情感任务结果更有代表性，可选情感。

### 图 3：Single vs Multi

展示：

- 完整率
- 事实一致率
- 工具成功率
- 延迟

如果量纲差异太大，可用表格 + 简单双图，不必为了“必须一张图”做复杂可视化。

#### 3. 写 `results/final_numbers.md`

至少包含：

- 表 1
- 表 2
- QLoRA 相对 Base 的提升
- 1～3 个典型失败案例
- 数据域声明
- 测试样本规模
- Agent Benchmark 数量
- 简历可用数字
- 面试一句话解释
- 每个数字的来源文件和适用范围
- fixture Benchmark 与真实线上/实时数据指标的边界
- 历史性能提升所依赖的同协议前后日志
- 禁止使用或尚待验证的历史数字

### 验收标准

- [ ] Risk 模型对比与 Agent 消融结果可复现
- [ ] Sentiment 未完成时已明确标为未运行
- [ ] README 已明确实验协议
- [ ] 简历数字没有使用存在泄漏的数据
- [ ] 面试时可以用两张表解释“为什么微调”和“为什么 Multi-Agent”
- [ ] fixture Benchmark 指标已明确限定为固定 30 题、冻结数据和统一运行配置
- [ ] 没有同场景、同模型、同环境前后日志时，未写“2 分钟+ → 约 90 秒”
- [ ] 未将弱标注数据表述为 10 万条人工标注

---

# 5. 两张最终结果表

## 表 1：模型效果（证据链 ①）

> 同一任务内三种模型必须基于同一 `eval_test.csv`。Risk 为 P0；Sentiment 未完成时对应列统一填写 `未运行`，不得用非同协议旧结果补位。

| Model | Risk Macro-F1 | Risk MAE | Sentiment Macro-F1 | Sentiment MAE |
|---|---:|---:|---:|---:|
| TF-IDF + LR | XX | XX | XX | XX |
| Qwen3-8B Base | XX | XX | XX | XX |
| Qwen3-8B + QLoRA | **XX** | **XX** | **XX** | **XX** |

附加：

- Accuracy
- Confusion Matrix
- Valid Output Rate（Base / QLoRA 诊断用）

QLoRA 相对提升：

\[
RelativeImprovement=
\frac{F1_{QLoRA}-F1_{Base}}
{F1_{Base}}\times100\%
\]

必须同时记录绝对提升：

```text
Macro-F1: 0.XX → 0.XX
绝对提升：+X.XX
相对提升：+XX.X%
```

---

## 表 2：Agent 消融（证据链 ②）

| Architecture | 完整率 | 事实一致率 | 工具成功率 | P50 延迟 | P95 延迟 |
|---|---:|---:|---:|---:|---:|
| Single-Agent | XX% | XX% | XX% | XXs | XXs |
| 5-Agent | XX% | XX% | XX% | XXs | XXs |

注意：

- 表中指标只适用于固定 30 题、冻结 fixtures 和统一运行配置，不代表线上全量或实时数据表现

- 不预设 5-Agent 一定更优
- 若 5-Agent 完整性更高但延迟更差，应如实呈现收益 / 成本权衡
- 失败任务必须保留在分母

---

# 6. 实验失败时的降级方案

## 6.1 QLoRA 重训失败

### 情况 A：旧 Adapter 训练证据完整，且与新 Test / eval_test 无 `sample_id` 交集

只有数据文件哈希、代码版本、切分规则和旧训练 `sample_id` 均可重建时，才可以正式使用旧 Adapter。

### 情况 B：证据缺失、存在交集，或重训失败

旧 Adapter 结果：

- 可用于调试或 exploratory result
- **不得进入正式表 1**
- **不得作为简历 QLoRA 提升数字**

此时简历只保留可以被真实实验支持的 Agent 线或其他已验证内容。

---

## 6.2 弱标签抽查发现明显问题

不要只人工修改抽查出来的问题样本。

处理规则：

- 少量偶发错误：记录局限，继续
- 某一标签等级系统性异常：暂停并检查标签生成逻辑
- 某一任务整体明显不可信：该任务不进入正式模型证据链

宁可只保留 Risk 或 Sentiment 一个可靠任务，也不要凑两组数字。

---

## 6.3 LR 内存不足

在 Train 内固定 seed 抽样：

```text
≤ 100,000 条
```

Val / eval_test 不变。

必须记录实际训练样本数。

---

## 6.4 Qwen 推理耗时过长

不缩减三模型之间的评测公平性。

默认即使用 500 条。若吞吐测试证明时间充足，可在任何正式模型结果生成前统一扩至 1000 条。无论采用哪一规模，都必须：

- 一次性冻结
- 三个模型全部使用相同的新 eval_test
- 记录样本规模

不要 Base 跑 500、QLoRA 跑 1000，也不要看到结果后改变规模。

---

## 6.5 MCP 实时数据不可用

Benchmark 正式运行只使用已经冻结的 fixtures。

若某股票在 Benchmark 冻结之前无法获得完整 fixture，可以替换问题或股票。

**一旦 `questions.jsonl` 正式冻结，后续崩溃、超时、异常不得删除题目。**

---

## 6.6 Agent 正式运行受阻

正式规模固定为 **30 条 P0 Benchmark**，不扩到 40 条。5 条 smoke test 必须在正式冻结前暴露 fixture、评分器和 runner 问题。

一旦 30 题冻结，不得在看到结果后删题或替换题目；基础设施故障允许按预先定义的重跑规则重跑，但必须保留原失败日志。

---

## 6.7 Multi-Agent 没有优势

如实报告。

可能得到：

```text
Multi-Agent：
完整率更高
事实一致率相近
延迟更高
```

或者：

```text
Single-Agent 整体更优
```

都属于有效实验结果。

简历可改写为：

> 对 Single-Agent 与 5-Agent 开展架构消融，量化分析报告完整性、事实一致性及延迟的收益/成本权衡。

---

# 7. 项目停止条件

满足以下任一条件即停止扩展：

## 7.1 正常停止

Day 1 已完成验收，Risk 三模型正式表和 Agent 表 2 已完成，`final_numbers.md`、README 和简历数字已整理。Sentiment 不作为停止条件。

→ **停止优化。**

## 7.2 时间停止

第 7 天仅用于 QLoRA 重训、基础设施失败重跑与收尾；结束时无论 Sentiment 是否完成，都停止并基于已有可信结果收尾。

## 7.3 范围停止

任何新增：

- 模型
- Agent
- 指标
- 数据源
- 新框架
- 新检索模块
- 新可视化系统

默认不做。

---

# 8. README 与简历最终输出模板

## 8.1 README

```text
1. 项目介绍
2. 系统架构
3. 金融新闻风险 / 情感建模
4. Model Benchmark
5. Agent Ablation
```

实验部分必须明确：

- 新闻数据是弱标注数据，除非有真实人工标注记录，否则不使用“人工标注约 10 万条”
- fixture Benchmark 指标的固定样本与冻结数据适用范围
- 历史时延提升仅在存在同场景、同模型、同环境前后日志时使用

- 数据量
- 训练样本量
- eval_test 样本量
- 数据切分方式
- Base / QLoRA 控制变量
- Benchmark 问题数量
- Single / Multi 控制变量
- 数据域差异

---

## 8.2 简历模板

### 模型线

仅当无完全重复样本泄漏、旧 Adapter 可追溯或已按新切分重训，且正式实验完成后使用：

> 清洗并构建纳斯达克英文金融新闻弱标注数据，形成风险/情感 1～5 级任务，采用时间切分与精确去重排除完全重复样本，并降低事件泄漏风险；对 Qwen3-8B 开展 QLoRA 微调，风险 Macro-F1 达 0.XX，较 Base Model 绝对提升 X.XX / 相对提升 XX%，MAE 降至 X.XX，并与 TF-IDF+LR 基线对比。

如果风险 / 情感只有一个任务可信，只写一个，不强行同时写两个。

### Agent 线

> 基于 LangGraph 构建基本面、技术面、估值、新闻与汇总 5-Agent A 股投研系统；在 30 条固定 Benchmark 上与 Single-Agent 开展消融，在相同 LLM、工具和冻结数据下，报告完整率由 XX% 提升至 XX%，事实一致率 XX%，工具成功率 XX%，并量化 P50/P95 延迟开销。

如果 Multi-Agent 没有明显优势：

> 在 30 条固定 Benchmark 上开展 Single-Agent / 5-Agent 架构消融，量化报告完整性、事实一致性、工具成功率及 P50/P95 延迟的收益/成本权衡。

---

# 9. P0 执行清单

## Day 1

- [ ] 明确 Risk / Sentiment 的 1～5 标签语义
- [ ] URL / title / title+summary 精确去重
- [ ] 生成稳定 `sample_id`
- [ ] 按日期边界切 Train / Val / Test
- [ ] 检查 Train / Val / Test 的 sample_id 无交集
- [ ] 从 Test 固定分层抽取 `eval_test`
- [ ] 输出 `split_manifest.json`
- [ ] 标签分层抽查约 100 条
- [ ] 输出 `split_report.md` 与 `label_spotcheck.md` 并验收
- [ ] manifest 的 source/script/CSV/sample_ids 哈希完整
- [ ] 两个 `eval_test` 均固定为 500 条

## Day 2

- [ ] `eval/metrics.py`
- [ ] `eval/parser.py`
- [ ] TF-IDF + LR
- [ ] C ∈ {0.5,1,2} 仅在 Val 选优
- [ ] LR 在正式 `eval_test` 上输出结果

## Day 3～4

- [ ] 统一 Prompt
- [ ] Qwen3-8B Base
- [ ] 默认从新 Train 重训 1～2 万条
- [ ] 仅在训练证据完整时重建旧 Adapter 训练 `sample_id`
- [ ] 旧 Adapter 证据缺失或存在交集时未进入正式结果
- [ ] Base / QLoRA 使用相同 eval_test / parser / decoding
- [ ] Risk 表 1 完成；Sentiment 未完成时明确标为未运行

## Day 4

- [ ] 5 条 smoke test 通过且不计入正式结果
- [ ] 建立并冻结 30 条 P0 Benchmark
- [ ] 每题定义 `required_sections`
- [ ] 每题按需要定义 `required_facts`
- [ ] 股票跨多个行业
- [ ] 冻结全部 Benchmark 所需外部数据
- [ ] Tool 接口保持不变，只替换 Fixture Backend
- [ ] 评分规则和事实容差已在正式运行前冻结
- [ ] 正式输出后未修改题目、样本、评分规则或事实容差

## Day 5

- [ ] Single-Agent 正式运行
- [ ] 5-Agent 正式运行
- [ ] 所有失败任务保留在分母
- [ ] 完整率
- [ ] 事实一致率
- [ ] 工具成功率
- [ ] P50 / P95（标注 N=30，仅作描述性结果）
- [ ] 记录 LLM 与工具调用次数及预算上限
- [ ] 明确标注为系统级架构消融
- [ ] 表 2 完成

## Day 6

- [ ] README 5 部分
- [ ] 模型 F1 图
- [ ] 代表性 Confusion Matrix
- [ ] Agent 对比图 / 表
- [ ] `final_numbers.md`
- [ ] 数据域声明
- [ ] 简历数字
- [ ] 两条面试口径

---

# 10. P1 清单（默认全部不做）

- [ ] embedding / MinHash / SimHash 相似新闻聚类
- [ ] 事件级去重
- [ ] 200～500 条人工 Gold Dataset
- [ ] 双人标注
- [ ] Cohen's / Weighted Kappa
- [ ] QWK
- [ ] Bootstrap
- [ ] 显著性检验
- [ ] RMSE
- [ ] Token Cost
- [ ] 60～100 条 Agent Benchmark
- [ ] 大规模重复稳定性实验
- [ ] 自动通用事实核验
- [ ] 大范围 QLoRA 调参
- [ ] 新模型 / 新 Agent / 新工具

---

# 11. 最终执行优先级

如果时间继续被压缩，严格遵守：

> **Day 1 完成 → Risk 正式三模型对比 → Agent Benchmark → `final_numbers.md` / README / 简历 → Sentiment（有余力再做）**

最低可接受结果是：

1. Day 1 全部正式 CSV、manifest、report、spotcheck 生成并验收
2. Risk 的 TF-IDF+LR / Qwen3-8B Base / QLoRA 同协议正式对比
3. 30 条固定 Benchmark 的 Single-Agent / 5-Agent 系统级对比
4. `final_numbers.md` 记录每个数字的来源与适用范围
5. README 与简历使用可追溯的新数字

---

# 12. 变更记录

- 2026-08-28：补齐 Day 1 标准字段、标准化、股票任务粒度去重、冲突标签、固定 sample_id、500 条 eval_test 和 manifest 哈希协议；增加历史简历数字处理表，并统一 Sentiment 与 Agent 评分冻结优先级。

- 2026-08-26：将本版冻结为后续唯一执行基线；明确 Day 1 规则与完成状态；最低成功线收敛为 Day 1 + Risk 三模型 + Agent Benchmark + 成果整理；Sentiment 降为最后可选项；增加评分规则前置冻结、弱标注和 fixture 指标适用范围约束。

- 2026-08-24：按可行性审查收紧执行基线：Risk 设为最低模型成功线，eval_test 默认 500；旧 Adapter 默认重训；Agent 固定 30 题并增加 5 题 smoke test、公平性预算、评分容差和系统级架构消融结论边界；Day 7 仅作失败重跑与收尾缓冲。

| 日期 | 变更 | 理由 |
|---|---|---|
| 2026-08-24 | 建立原始 6 天执行基线 | 秋招倒计时，压缩为两条证据链 |
| 2026-08-24 | Adapter 泄漏检查改为 sample_id 交集检查 | 日期重叠不能直接证明样本泄漏 |
| 2026-08-24 | Test 冻结改为 manifest/hash，不强制提交大型 CSV | 避免 Git 仓库膨胀并提高可复现性 |
| 2026-08-24 | 新增固定 eval_test | 控制 Qwen3-8B 推理工作量并保证三模型同样本比较 |
| 2026-08-24 | 时间切分改为日期边界 | 避免同一天新闻跨数据集 |
| 2026-08-24 | Agent 完整率改为 required_sections | 不同类型问题需要不同必需模块 |
| 2026-08-24 | MCP fixtures 改为 Fixture Tool Backend | 保留 Agent 工具选择与调用能力，保证架构消融公平 |
| 2026-08-24 | 失败/超时任务禁止从分母剔除 | 防止高估 Agent 系统表现 |
| 2026-08-24 | 30 条 Agent Benchmark 设为 P0 | 进一步降低秋招优化时间成本 |
| 2026-08-24 | 增加统一 parser | 防止 Base / QLoRA 输出解析口径不一致 |

> 从此版本开始，除非直接影响两条证据链，否则不再新增优化项。
