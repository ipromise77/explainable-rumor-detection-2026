# LLM 置信度阈值优化实验

## 1. 实验背景

在前序实验中，我们建立了基于置信度的双层架构：本地基座模型（TF-IDF + LR）先预测，对于不确定样本（概率落在 [0.30, 0.70] 区间）调用 DeepSeek 大模型复核。但实际效果不理想——大模型很少真正推翻本地模型的判断。

分析混淆矩阵发现，验证集上有 37 个谣言被漏报（FN=37），而大模型仅救回了 1 个。核心原因是：**override 置信度阈值设得太高（0.85）**，大模型即使判断为谣言，但只给出 0.65~0.75 的置信度时，就不会触发翻转。

## 2. 优化策略

### 2.1 降低置信度阈值

将 LLM override 的置信度阈值从 **0.85 降低到 0.65**。

理论依据：分析已有 LLM 缓存发现，大模型对那些"伪装成新闻"的谣言往往给出 0.65~0.75 的置信度——它能识别出谣言特征，但不够"自信"。降低门槛可以让这些判断生效。

### 2.2 综合两路 LLM 缓存

我们有两套 LLM 缓存数据：

| 缓存 | 覆盖范围 | 样本数 |
|------|---------|--------|
| `llm_cache.jsonl` | prob ∈ [0.30, 0.70] 的摇摆样本 | 108 条 |
| `fn_recall_review_candidates.csv` | pred=0 且 prob ∈ [0.20, 0.50] 的疑似漏报样本 | 79 条 |

两套缓存有部分重叠，但也有互补。综合使用可以覆盖更多潜在的漏报样本。

### 2.3 组合策略

```
路径1: 若 prob ∈ [0.30, 0.70] 且有 v1 缓存
       若 local_pred=0 且 LLM 判断为谣言 且 confidence >= 0.65
       → 翻转为谣言

路径2: 若 local_pred=0 且 prob ∈ [0.15, 0.50] 且有 v2 缓存
       若 LLM 判断为谣言 且 confidence >= 0.65
       → 翻转为谣言
```

## 3. 实验结果

### 3.1 整体效果

| 指标 | Baseline | 优化后 | 变化 |
|------|----------|--------|------|
| **Accuracy** | 88.03% | **88.78%** | **+0.75%** |
| **Macro-F1** | 0.8757 | **0.8843** | +0.86% |
| **FN（漏报）** | 37 | **32** | **-5** |
| **FP（误报）** | 11 | 13 | +2 |

混淆矩阵变化：

```
Baseline:              Optimized:
[[215, 11],      →     [[213, 13],
 [37, 138]]             [32, 143]]
```

### 3.2 翻转详情

共触发 7 次翻转操作：

| 状态 | 样本概率 | LLM置信度 | 来源 | 文本摘要 |
|------|---------|-----------|------|---------|
| 救回 FN | 0.388 | 0.65 | v1 | Here's the police report. Somehow #Ferguson cops... |
| 救回 FN | 0.318 | 0.65 | v1 | Line of police cars with high beams on greets... |
| 救回 FN | 0.204 | 0.85 | v2 | Disgusting: MO chapter of Klan raising money... |
| 救回 FN | 0.395 | 0.72 | v1 | Total number of people who have left the cafe... |
| 救回 FN | 0.268 | 0.70 | v2 | Here are the 3 locations of shootings in #Ottawa... |
| 新增 FP | 0.330 | 0.75 | v1 | "No survivors" from #Germanwings crash... |
| 新增 FP | 0.432 | 0.75 | v2 | VIDEO: Key moments in today's Parliament Hill... |

### 3.3 分析

**救回的 5 个 FN 样本**：这些样本的共同特征是包含煽动性或未经证实的表述（如 "Somehow cops valued..."、"Klan raising money as reward"），LLM 能识别出谣言特征但置信度不够高（0.65~0.85）。

**新增的 2 个 FP 样本**：都是关于突发事件的真实报道（Germanwings 空难、Ottawa 枪击案），LLM 误判的原因是这些文本的表述方式与谣言样本高度相似。这是降低阈值的代价。

**净收益**：救回 5 个 FN，牺牲 2 个 FP，净增 3 个正确预测。

## 4. 剩余 FN 样本分析

优化后仍有 32 个 FN 样本无法救回。分析发现：

1. **LLM 也判不出来**：这 32 个样本中有 16 个已经过 LLM 复核，但 LLM 也给出了 `label=0`（非谣言），且置信度高达 0.85~0.95

2. **极度伪装**：这些样本的文本看起来完全像真实新闻报道，例如：
   - "Our thoughts and prayers go out to Nathan Cirillo who died today..."
   - "Stretchers taken from Sydney cafe after police storm building..."

3. **信息熵极限**：这些样本从字面信息上已经无法区分真假，需要外部事实核查才能判定

## 5. 结论

1. **降低置信度阈值是有效的**：从 0.85 降到 0.65，在可接受的 FP 增加范围内（+2），显著减少了 FN（-5）

2. **组合策略优于单一策略**：综合两路 LLM 缓存能覆盖更多边界样本

3. **存在优化上限**：剩余的 32 个 FN 样本是"硬骨头"，需要外部知识才能突破，单纯依靠文本特征和 LLM 已达瓶颈

## 6. 复现方式

运行优化评估脚本：

```bash
python scripts/run_optimized_evaluation.py
```

输出文件：
- `results/optimized_predictions.csv` - 详细预测结果
- `results/optimized_metrics.json` - 评估指标汇总
