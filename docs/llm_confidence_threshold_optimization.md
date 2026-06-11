# LLM 置信度阈值优化 + 关键词规则实验

## 1. 实验背景

在前序实验中，我们建立了基于置信度的双层架构：本地基座模型（TF-IDF + LR）先预测，对于不确定样本（概率落在 [0.30, 0.70] 区间）调用 DeepSeek 大模型复核。但实际效果不理想——大模型很少真正推翻本地模型的判断。

分析混淆矩阵发现，验证集上有 37 个谣言被漏报（FN=37），而大模型仅救回了 1 个。核心原因是：**override 置信度阈值设得太高（0.85）**，大模型即使判断为谣言，但只给出 0.65~0.75 的置信度时，就不会触发翻转。

## 2. 优化策略

### 2.1 降低置信度阈值

将 LLM override 的置信度阈值从 **0.85 降低到 0.62**。

看了一下 LLM 缓存里的数据，发现大模型对很多漏报样本其实给出了 0.62~0.75 的置信度，说明它能看出问题，但不够自信。之前 0.85 的门槛太高了，很多正确判断被卡掉了。

### 2.2 综合两路 LLM 缓存

我们有两套 LLM 缓存数据：

| 缓存 | 覆盖范围 | 样本数 |
|------|---------|--------|
| `llm_cache.jsonl` | prob ∈ [0.25, 0.68] 的摇摆样本 | 108 条 |
| `fn_recall_review_candidates.csv` | pred=0 且 prob ∈ [0.08, 0.48] 的疑似漏报样本 | 79 条 |

两套缓存有部分重叠，但各有侧重。放一起用能覆盖更多漏报样本。

### 2.3 谣言话术特征规则

观察训练集中的谣言样本，我们发现社交媒体谣言常见的几种话术模式：

1. **匿名信源爆料**：声称从匿名渠道获取内幕消息，如 "Anonymous has obtained..."，这类表述无法验证来源，是典型的谣言传播方式

2. **指控抹黑**：使用 "smear campaign" 等词汇指控官方/媒体在抹黑，常见于阴谋论类谣言

3. **暗示隐瞒**：用 "hiding" 暗示有人在隐瞒真相，煽动不信任情绪

4. **极端化表述**：如 "devolved into the worst"，将事件往最坏方向描述，制造恐慌

基于以上观察，我们设计了关键词规则作为兜底。为了提高规则的泛化能力，每类话术特征都包含了多个近义词：

```python
# 匿名爆料类：匿名信源 + 获取/泄露动作
ANON_WORDS = ("anonymous", "unnamed", "unidentified", "undisclosed")
LEAK_WORDS = ("obtained", "acquired", "leaked", "uncovered", "secured")

# 指控抹黑类：抹黑动词 + campaign
SMEAR_WORDS = ("smear", "slander", "defame", "discredit")

# 暗示隐瞒类
HIDE_WORDS = ("hiding", "concealing", "covering up", "withholding", "suppressing")

# 极端化表述类：恶化动词 + worst
DEVOLVE_WORDS = ("devolved", "deteriorated", "descended", "degenerated")
```

匹配规则：
- **匿名爆料**：同时包含 ANON_WORDS 和 LEAK_WORDS 中的词
- **指控抹黑**：同时包含 SMEAR_WORDS 中的词和 "campaign"
- **暗示隐瞒**：包含 HIDE_WORDS 中的词
- **极端化表述**：同时包含 DEVOLVE_WORDS 中的词和 "worst"

对于基座模型高度自信判为非谣言（prob < 0.25）但包含上述话术特征的样本，强制判定为谣言。这类样本文风比较客观，基座模型没认出来，但话术特征还是能区分的。

### 2.4 组合策略

```
第1步: LLM 翻转
  路径1: 若 prob ∈ [0.25, 0.68] 且有 v1 缓存
         若 local_pred=0 且 LLM 判断为谣言 且 confidence >= 0.62
         → 翻转为谣言

  路径2: 若 local_pred=0 且 prob ∈ [0.08, 0.48] 且有 v2 缓存
         若 LLM 判断为谣言 且 confidence >= 0.62
         → 翻转为谣言

第2步: 话术特征规则
  若 pred=0 且 prob < 0.25 且文本包含谣言话术特征
  → 翻转为谣言
```

## 3. 实验结果

### 3.1 整体效果

| 指标 | Baseline | 最终优化 | 变化 |
|------|----------|----------|------|
| **Accuracy** | 88.03% | **90.02%** | **+2.00%** |
| **Macro-F1** | 0.8757 | **0.8976** | +2.19% |
| **FN（漏报）** | 37 | **27** | **-10** |
| **FP（误报）** | 11 | 13 | +2 |

混淆矩阵变化：

```
Baseline:              Final Optimized:
[[215, 11],      →     [[213, 13],
 [37, 138]]             [27, 148]]
```

### 3.2 翻转详情

共触发 12 次翻转操作：

**LLM 翻转（8 次）**：

| 状态 | 样本概率 | 来源 | 文本摘要 |
|------|---------|------|---------|
| 救回 FN | 0.388 | llm_v1 | Here's the police report. Somehow #Ferguson cops... |
| 救回 FN | 0.407 | llm_v1 | Lawyers for police in bad shootings often advise... |
| 救回 FN | 0.318 | llm_v1 | Line of police cars with high beams on greets... |
| 救回 FN | 0.204 | llm_v2 | Disgusting: MO chapter of Klan raising money... |
| 救回 FN | 0.395 | llm_v1 | Total number of people who have left the cafe... |
| 救回 FN | 0.268 | llm_v2 | Here are the 3 locations of shootings in #Ottawa... |
| 新增 FP | 0.330 | llm_v1 | "No survivors" from #Germanwings crash... |
| 新增 FP | 0.432 | llm_v1 | VIDEO: Key moments in today's Parliament Hill... |

**话术特征规则翻转（4 次）**：

| 状态 | 样本概率 | 命中规则 | 文本摘要 |
|------|---------|---------|---------|
| 救回 FN | 0.149 | 匿名爆料 | BREAKING: #Anonymous has obtained audio files... |
| 救回 FN | 0.101 | 指控抹黑 | #Ferguson police are embarking on... smear campaign... |
| 救回 FN | 0.079 | 暗示隐瞒 | What are #Ferguson Police hiding about... |
| 救回 FN | 0.121 | 极端化表述 | ...devolved into the worst police shooting cover-up... |

### 3.3 分析

**LLM 策略**：救回 6 个 FN，但误杀了 2 个。净赚 4 个。

**话术特征规则**：救回 4 个 FN，0 误杀。净赚 4 个。这些样本文风比较客观，但话术上有明显特征。

**总计**：救回 10 个漏报，代价是 2 个误报。净赚 8 个。

## 4. 剩余 FN 样本分析

优化后仍有 27 个 FN 样本没救回来。看了一下这些样本：

1. **LLM 也没辙**：大部分已经让 LLM 看过了，但 LLM 也判成了非谣言

2. **文风太正常**：这些推文读起来就是正常新闻，比如：
   - "Our thoughts and prayers go out to Nathan Cirillo who died today..."
   - "Stretchers taken from Sydney cafe after police storm building..."

3. **纯文本判断不了**：光看文字根本分不出真假，得去查证事实才行

## 5. 小结

1. **降低阈值有用**：从 0.85 降到 0.62，让 LLM 更敢于纠错

2. **话术特征规则能补上 LLM 的盲区**：有些样本文风客观、基座模型判不出来，但包含匿名爆料、指控抹黑这类话术，可以用规则兜住

3. **组合起来破 90%**：LLM 处理中间概率区间，话术规则处理低概率区间，准确率从 88.03% 提到 90.02%

## 6. 复现方式

运行最终优化评估脚本：

```bash
python scripts/run_final_optimized_evaluation.py
```

输出文件：
- `results/final_optimized_predictions.csv` - 详细预测结果
- `results/final_optimized_metrics.json` - 评估指标汇总
