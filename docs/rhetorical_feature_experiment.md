# 谣言话术辅助特征实验

## 1. 实验目的

`feat/llm-override-threshold-optimization` 分支中提出了若干谣言话术模式，例如：

- 匿名信源爆料类：`anonymous/unnamed/unidentified/undisclosed` + `obtained/acquired/leaked/uncovered/secured`
- 指控抹黑类：`smear/slander/defame/discredit` + `campaign`
- 暗示隐瞒类：`hiding/concealing/covering up/withholding/suppressing`
- 极端化表述类：`devolved/deteriorated/descended/degenerated` + `worst`

这些模式可用于解释，但如果直接写成规则覆盖标签，容易形成针对 `val.csv` 的手工调参。为避免验证集过拟合，本实验将它们作为辅助特征交给逻辑回归学习权重，并只用 `train_clean.csv` 的 5-fold 交叉验证判断是否保留。

## 2. 实验设置

运行：

```bash
python scripts/run_rhetorical_feature_experiment.py
```

脚本比较三类方案：

1. baseline：原三路 TF-IDF + Logistic Regression 集成。
2. strict：只加入 4 个组合话术特征及其计数。
3. component：加入话术组件特征，例如 `anonymous_word_any`、`campaign_word`、`worst_word` 等，再由模型学习权重。

每个话术特征版本测试 scale = `0.5, 1.0, 2.0, 4.0`，选择依据只使用 `train_clean.csv` 的 5-fold 结果，`val.csv` 只做最终一次对照。

## 3. 训练集覆盖情况

严格组合特征在训练集中非常稀疏：

| feature | train active | rumor | non-rumor |
| --- | ---: | ---: | ---: |
| anonymous_leak_signal | 0 | 0 | 0 |
| smear_campaign_signal | 0 | 0 | 0 |
| concealment_signal | 1 | 1 | 0 |
| extreme_worst_signal | 0 | 0 | 0 |

组件特征覆盖稍多，但先验并不稳定：

| feature | train active | rumor | non-rumor | rumor rate |
| --- | ---: | ---: | ---: | ---: |
| anonymous_word_any | 8 | 3 | 5 | 0.375 |
| smear_word_any | 6 | 3 | 3 | 0.500 |
| campaign_word | 5 | 0 | 5 | 0.000 |
| devolve_word_any | 3 | 2 | 1 | 0.667 |
| worst_word | 5 | 1 | 4 | 0.200 |

这说明这些词并不是训练集中稳定的谣言先验。它们在 `val.csv` 中命中的几条样本刚好大多是谣言，但不能据此手写规则，否则会有验证集调参风险。

## 4. 5-fold 结果

| strategy | Accuracy mean | Macro-F1 mean | Rumor F1 mean | Recall(1) mean |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.8641 | 0.8606 | 0.8385 | 0.8101 |
| strict scale=0.5 | 0.8641 | 0.8606 | 0.8385 | 0.8101 |
| strict scale=1.0 | 0.8641 | 0.8606 | 0.8385 | 0.8101 |
| strict scale=2.0 | 0.8638 | 0.8602 | 0.8380 | 0.8093 |
| strict scale=4.0 | 0.8634 | 0.8598 | 0.8375 | 0.8084 |
| component scale=2.0 | 0.8627 | 0.8591 | 0.8367 | 0.8076 |

根据训练集内部 5-fold，最佳话术特征方案与 baseline 完全持平，没有证明出稳定收益。

## 5. Val 对照

按 5-fold 选择 strict scale=0.5 后，在 `val.csv` 上结果如下：

| 模型 | Accuracy | Macro-F1 | Rumor F1 | FP | FN | 预测变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.8803 | 0.8757 | 0.8519 | 11 | 37 | - |
| rhetorical features | 0.8803 | 0.8757 | 0.8519 | 11 | 37 | 0 |

## 6. 结论

本实验没有带来准确率提升。原因主要有两点：

1. 严格组合话术特征在训练集中几乎不出现，模型无法学习稳定权重。
2. 更宽松的组件特征在训练集中并不稳定，例如 `campaign` 和 `worst` 多数对应非谣言，不能简单当作谣言先验。

因此，当前不建议将这些话术特征并入最终分类模型，也不建议用它们直接覆盖标签。更合理的用法是：在解释阶段把模型已经命中的高贡献词归纳为“匿名爆料”“指控抹黑”“暗示隐瞒”“极端化表述”等证据模式，从而增强解释可读性，而不改变最终分类决策。
