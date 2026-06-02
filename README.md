# 可解释的谣言检测

本项目完成《人工智能导论》大作业“可解释的谣言检测”：输入一条推文文本，输出二分类结果，其中 `0` 表示非谣言，`1` 表示谣言，并给出判断依据。项目包含本地可复现的 TF-IDF 集成检测器，以及可选的学校 API 大模型解释增强模块。

## 项目结构

```text
.
├── rumer2026/
│   ├── train.csv
│   └── val.csv
├── src/
│   ├── rumor_detector.py   # 核心检测类、预处理、解释生成
│   ├── llm_enhanced.py      # 学校API大模型增强检测与解释
│   ├── train.py            # 训练并保存模型
│   ├── evaluate.py         # 评估保存后的模型
│   ├── evaluate_llm.py      # 评估LLM增强模式
│   └── predict.py          # 单条文本预测
├── models/
│   └── rumor_ensemble.joblib
├── results/
│   ├── metrics.json
│   └── val_predictions.csv
├── report.pdf
└── requirements.txt
```

## 环境安装

建议使用 Python 3.11。安装依赖：

```bash
pip install -r requirements.txt
```

## 训练

在项目根目录运行：

```bash
python -m src.train --with-explanations
```

脚本会读取 `rumer2026/train.csv` 与 `rumer2026/val.csv`，训练模型并生成：

- `models/rumor_ensemble.joblib`：保存后的检测模型
- `results/metrics.json`：验证集指标
- `results/val_predictions.csv`：验证集预测结果和解释

## 评估

```bash
python -m src.evaluate --show-examples 3
```

当前模型在 `val.csv` 上的结果：

| 指标 | 数值 |
| --- | ---: |
| Accuracy | 0.8803 |
| Macro-F1 | 0.8757 |
| Rumor F1 | 0.8519 |

混淆矩阵 `[[TN, FP], [FN, TP]]` 为：

```text
[[215, 11],
 [ 37, 138]]
```

## 单条文本预测

```bash
python -m src.predict "BREAKING: reports say the city confirms the story is false after investigation #news"
```

## LLM 增强模式

先在 `.env` 中填写学校 API：

```env
SJTU_API_KEY=你的学校API_KEY
SJTU_BASE_URL=https://models.sjtu.edu.cn/api/v1
SJTU_MODEL=deepseek-reasoner
```

单条增强预测：

```bash
python -m src.llm_enhanced --force-llm "input tweet text"
```

全量验证时，推荐使用“低置信度复核 + 高置信覆盖”策略：

```bash
python -m src.evaluate_llm --low 0.30 --high 0.70 --allow-override --override-confidence 0.85 --quiet
```

该策略只对本地模型谣言概率位于 `[0.30, 0.70]` 的样本调用大模型。当前验证结果：

| 模式 | Accuracy | Macro-F1 | Rumor F1 | LLM调用 |
| --- | ---: | ---: | ---: | ---: |
| 本地TF-IDF集成 | 0.8803 | 0.8757 | 0.8519 | 0 |
| LLM增强高置信覆盖 | 0.8828 | 0.8787 | 0.8563 | 70 |

默认增强模式不会让大模型随意改标签，而是优先使用本地模型标签，由大模型润色解释；只有当大模型与本地模型冲突且自评置信度不低于 `0.85` 时才覆盖。

输出示例：

```json
{
  "label": 1,
  "prob_rumor": 0.8598,
  "explanation": "预测为1（谣言），平均置信度0.860。主要支持谣言的证据包括：..."
}
```

也可以在代码中直接调用课程要求的检测类：

```python
from src.rumor_detector import RumourDetectClass

detector = RumourDetectClass("models/rumor_ensemble.joblib")
label = detector.classify("input tweet text")
reason = detector.explain("input tweet text")
```

## 方法说明

基础模型采用三路轻量线性集成：

- 词级 TF-IDF：捕捉关键词、二元词组和部分三元词组。
- 字符级 TF-IDF：提升对 hashtag、拼写变体、短文本片段的鲁棒性。
- 逻辑回归集成：平均三个互补模型的谣言概率，兼顾准确率、运行速度和可解释性。

基础解释模块计算当前文本中每个 TF-IDF 特征与逻辑回归权重的乘积，汇总为“支持谣言”和“支持非谣言”的局部证据。LLM 增强模块会把本地证据和相似训练样本发送给 `deepseek-reasoner`，让大模型生成更自然的中文判断依据，并在极少数高置信冲突样本上进行保守修正。

## 创新点

1. 多视角可解释集成：同时利用词级语义线索和字符级局部模式，再把各子模型的局部贡献合并为自然语言解释。
2. 平台特征保留式预处理：URL、用户提及、hashtag 不简单删除，而是归一化为可学习标记，降低短文本噪声。
3. 低置信度 LLM 复核：只让 `deepseek-reasoner` 处理不确定样本，兼顾解释质量、准确率和运行时间。
4. RAG 式相似样本提示：调用 LLM 时提供相似训练样本和本地模型证据，使解释更贴近数据集分布。

## 备注

暂无
