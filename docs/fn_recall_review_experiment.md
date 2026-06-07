# 低置信 False Negative 召回复核实验

## 1. 实验动机

事件级错误分析显示，event 0 和 event 1 的主要问题不是误报，而是漏报：模型把一部分真实谣言预测为非谣言。这类错误对应 `false_negative`，会降低谣言类召回率。因此本轮实验不做全局阈值调整，而是设计一个更保守的复核策略：

```text
本地模型预测为 0（非谣言）
且 0.20 <= prob_rumor < 0.50
        ↓
调用 deepseek-reasoner 复核
        ↓
只有当 LLM 建议为 1 且置信度 >= 0.85 时，才覆盖为 1
```

该策略沿用项目已有的 `0.85` 高置信覆盖阈值，目的是尽量减少新增 false positive。

## 2. 运行方式

运行：

```bash
python scripts/run_fn_recall_review_experiment.py
```

脚本会读取 `.env` 中的学校 API 配置，并生成：

- `results/fn_recall_review_candidates.csv`
- `results/fn_recall_review_sweep.csv`
- `results/fn_recall_review_summary.json`

API 原始缓存写入 `results/fn_review_llm_cache.jsonl`，该文件已加入 `.gitignore`，不提交到仓库。

## 3. 候选样本

在 `val.csv` 上，候选规则筛出 79 条样本：

| 候选规则 | 样本数 | 真实谣言 | 真实非谣言 |
| --- | ---: | ---: | ---: |
| local_pred=0 且 0.20 <= prob_rumor < 0.50 | 79 | 21 | 58 |

这说明该区间确实包含可被救回的漏报样本，但非谣言样本更多，因此覆盖必须保守。

## 4. 实验结果

本地模型基线：

| Accuracy | Macro-F1 | Rumor F1 | Precision(1) | Recall(1) | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8803 | 0.8757 | 0.8519 | 0.9262 | 0.7886 | 11 | 37 |

低置信 false negative 复核策略：

| 策略 | Accuracy | Macro-F1 | Rumor F1 | Precision(1) | Recall(1) | 覆盖数 | 救回 FN | 新增 FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low=0.20, high=0.50, confidence>=0.85 | 0.8828 | 0.8784 | 0.8554 | 0.9267 | 0.7943 | 1 | 1 | 0 |

混淆矩阵从：

```text
[[215, 11],
 [ 37, 138]]
```

变为：

```text
[[215, 11],
 [ 36, 139]]
```

## 5. 关键样本

被最终策略救回的样本是：

| id | event | 原概率 | 原预测 | LLM复核 | LLM置信度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500328758201815041 | 1 | 0.2038 | 0 | 1 | 0.85 |

文本内容为：

```text
Disgusting: MO chapter of Klan raising money as “reward” for the officer killing #MikeBrown. #Ferguson #UniteBlue http://t.co/x...
```

LLM 给出的理由是该文本中 “Klan raising money as reward for the officer killing” 与相似训练样本中的谣言表述高度一致，因此建议覆盖为谣言。

## 6. 结论

本实验说明：面向 false negative 的定向 LLM 复核可以在不增加误报的情况下救回 1 条漏报样本，使 Accuracy 从 0.8803 提升到 0.8828，Rumor F1 从 0.8519 提升到 0.8554。

需要注意的是，这个提升幅度很小，且结果是在 `val.csv` 上观察到的。因此报告中应把它表述为“保守复核策略带来的小幅改进和创新点”，而不是夸大为大幅性能突破。它真正的价值在于：针对事件级错误分析发现的 false negative 问题，给出了一个可解释、可控、运行成本有限的复合模型改进方案。
