# 数据质量审计与方案迭代记录

## 1. 背景

老师提示数据集可能存在“不干净”的情况，例如文本相同但标签不同。由于本项目会对 URL、用户提及、hashtag 等社交媒体信息做归一化，如果两条推文只在短链接上不同，预处理后也可能变成同一个模型输入。因此我们对 `train.csv` 和 `val.csv` 做了数据质量审计。

## 2. 当前预处理规则

项目中的核心预处理位于 `src/rumor_detector.py` 的 `preprocess_text()`，审计脚本 `scripts/audit_data_quality.py` 使用相同规则：

1. HTML 反转义，例如 `&amp;` 还原为 `&`。
2. 全部转小写。
3. URL 统一替换为 `URLTOKEN`。
4. `@用户` 统一替换为 `USERTOKEN`。
5. `#话题` 展开为 `HASHTAG_话题 话题`，同时保留话题词本身。
6. 多个空白字符压缩为一个空格。

该设计不是简单删除社交媒体线索，而是将短链、用户提及和话题标签转换成模型可学习的稳定特征。

## 3. 审计结果

按当前预处理规则分组后，如果同一 `normalized_text` 同时对应 `label=0` 和 `label=1`，就判定为标签冲突。

| 数据集 | 冲突组数 | 涉及行数 | 说明 |
| --- | ---: | ---: | --- |
| `train.csv` | 3 | 7 | 存在同一预处理文本对应不同标签 |
| `val.csv` | 0 | 0 | 未发现同类标签冲突 |

详细冲突表已生成到：

- `results/train_preprocess_label_conflicts.csv`
- `results/val_preprocess_label_conflicts.csv`
- `results/data_quality_summary.json`

其中 `issue_no` 是异常样本序号，`row_index` 是 pandas 读取后的 0 基行号，`csv_line_no` 是原始 CSV 文件中的实际行号。

## 4. 典型问题样本

最典型的一组是完全相同文本但标签不同：

```text
DARREN WILSON MURDERED AN UNARMED TEEN LEFT HIS BODY IN THE STREET FOR 4 HRS Anything else is irrelevant. #MikeBrown #Ferguson #CoverUp
```

| id | label | 含义 |
| --- | ---: | --- |
| `500291056445845504` | 1 | 谣言 |
| `500396175753633792` | 0 | 非谣言 |

另外两组主要是短链接不同，但经过 URL 归一化后主体文本相同，导致同一个模型输入对应相反标签。

三组异常原因可概括为：

| 冲突组 | 涉及序号 | 异常原因 |
| --- | --- | --- |
| 1 | `issue_no=1,2` | 原始文本完全相同，但同一文本被分别标注为谣言和非谣言。 |
| 2 | `issue_no=3,4,5` | 文本主体完全相同，只是短链接不同；URL 归一化后输入相同，但标签冲突。 |
| 3 | `issue_no=6,7` | 文本主体和前两个链接相同，最后一个短链接不同；URL 归一化后输入相同，但标签冲突。 |

## 5. 关于 t.co 短链接的分析

异常组 2 和组 3 中出现的 `http://t.co/...` 是 X/Twitter 的官方短链接格式。`t.co` 后面的字符串，例如 `B4hWevO7l9`、`mUE41Zp4nA`、`A4RJS4unlx`，本质上是平台用于映射原始 URL 的不透明标识符。它们看起来像随机 token，通常不携带可直接解释的语义，不能仅凭字符串本身判断新闻类别、事件主题、谣言倾向或文本情感。

因此，在本项目中保留具体短链接字符串反而可能让模型记住偶然噪声，削弱泛化能力。把所有 URL 统一替换为 `URLTOKEN` 是合理的：模型仍能知道文本中出现了链接，但不会学习某个短链 token 的偶然编号。

不过，这也会暴露数据中的标签冲突：如果两条推文文本主体完全相同，只是短链接不同，那么经过 URL 归一化后它们会变成同一个模型输入。如果这些样本的标签一个是 `0`、一个是 `1`，模型训练时就会看到“同一个输入对应两个相反标签”。这类问题更应视为数据标注冲突或重复采样噪声，而不是短链接字符串本身有特殊含义。

具体例子如下。原始文本 A：

```text
Teenager #MikeBrown won't start college on Monday because he was shot ten times by a #Ferguson police officer. http://t.co/B4hWevO7l9
```

原始文本 B：

```text
Teenager #MikeBrown won't start college on Monday because he was shot ten times by a #Ferguson police officer. http://t.co/mUE41Zp4nA
```

两条文本的主体内容完全相同，只是最后的 `t.co` 短链接不同。经过本项目预处理后，它们都会变成：

```text
teenager HASHTAG_mikebrown mikebrown won't start college on monday because he was shot ten times by a HASHTAG_ferguson ferguson police officer. URLTOKEN
```

也就是说，从模型视角看，这两条样本已经是同一个输入。如果标签不同，就不是模型可以通过学习文本内容解决的问题，而是训练数据自身存在冲突。

该处理方式来自文本分类和社交媒体 NLP 中常见的归一化思想。课程支持文档中的逻辑回归参考代码使用了更基础的预处理：

```python
X_train = X_train.str.lower().str.replace('[^\w\s]', '', regex=True)
```

也就是“小写化 + 去标点”。该参考代码没有专门处理 URL，但大作业要求模型具有一定泛化能力。考虑到推文中大量存在随机短链接，直接保留 URL 会导致模型记忆无语义的链接字符串，因此本项目进一步将 URL 统一归一化为 `URLTOKEN`。这样既保留了“文本含链接”这一社交媒体信号，也减少了随机短链接对模型泛化能力和鲁棒性的干扰。

## 6. 清洗处理与重训结果

我们采用保守处理：保留原始 `train.csv` 不动，额外生成 `rumer2026/train_clean.csv`。清洗版训练集删除了 `results/train_preprocess_label_conflicts.csv` 中记录的 7 行冲突样本，其余样本保持原顺序和原始内容。

处理原则如下：

1. 不修改 `val.csv`，因为评分要求是在原始 `val.csv` 上计算准确率。
2. 不直接改标签，不做多数投票。冲突组本身标注不可判定，删除比主观改标签更稳健。
3. 训练脚本优先读取 `train_clean.csv`；如果该文件不存在，则回退到 `train.csv`。

清洗结果：

| 项目 | 数值 |
| --- | ---: |
| 原始训练集行数 | 2840 |
| 删除冲突样本 | 7 |
| 清洗后训练集行数 | 2833 |
| 验证集行数 | 401 |

清洗后重训结果如下：

| 训练数据 | Accuracy | Macro-F1 | Rumor F1 | 说明 |
| --- | ---: | ---: | ---: | --- |
| 原始 `train.csv` | 0.8803 | 0.8757 | 0.8519 | 作为对照 |
| 清洗 `train_clean.csv` | 0.8803 | 0.8757 | 0.8519 | 当前默认模型 |

进一步比较发现，原始模型和清洗模型在 `val.csv` 上的预测标签完全一致，变化条数为 0；谣言概率存在轻微变化，最大绝对变化约 0.0315，平均绝对变化约 0.0007。说明这 7 行冲突样本数量较少，对最终验证集 0/1 预测没有造成可见影响，但清洗后训练数据逻辑更一致，可解释性和数据质量说明更扎实。

相关文件：

- `rumer2026/train_clean.csv`
- `results/metrics_raw_train.json`
- `results/metrics_clean_train.json`
- `results/clean_vs_raw_comparison.json`

## 7. 后续方案迭代记录

在模型方案上，我们先实现了本地可复现的三路 TF-IDF + 逻辑回归集成模型。清洗训练集后，本地模型验证集准确率为 0.8803。随后接入学校 API 的 `deepseek-reasoner` 做低置信样本复核和解释生成。

测试发现，如果让大模型直接接管低置信样本标签，准确率反而下降；因此最终采用更稳健的策略：本地模型负责主要分类，大模型主要负责结合本地证据和相似训练样本生成中文解释，只有当大模型与本地模型冲突且自评置信度不低于 0.85 时才允许覆盖。该策略在 `val.csv` 上将准确率从 0.8803 小幅提升到 0.8828，同时提升了判断依据的自然性和可读性。

这说明我们的改进重点不是盲目堆叠大模型，而是围绕课程评分要求进行工程化取舍：准确率、解释性、可复现性和合理运行时间之间保持平衡。
