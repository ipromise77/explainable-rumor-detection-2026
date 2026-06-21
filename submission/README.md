# 最终提交清单

Canvas 提交内容为本 GitHub 仓库地址：

```text
https://github.com/ipromise77/explainable-rumor-detection-2026
```

提交前请确认：

- `README.md` 已说明安装、训练、评估、单条预测、demo 页面和报告编译方式。
- `report.tex` 已填入真实姓名、学号和贡献比例。
- `report.pdf` 已由 `python scripts/build_report.py` 或 XeLaTeX 编译生成，并提交到仓库。
- `python -m src.evaluate --show-examples 0` 可复现默认主模型 Accuracy `0.8803`。
- `python scripts/run_final_detector_evaluation.py` 可复现备选增强模型 Accuracy `0.8928`。
- `.env` 不提交；如需运行学校 API 解释增强，根据 `.env.example` 在本地填写。
- `results/llm_cache.jsonl` 和 `results/fn_review_llm_cache.jsonl` 不作为默认复现依赖。

人员信息待填写项：

- 组长：姓名、学号、贡献比例
- 成员 A：姓名、学号、贡献比例
- 成员 B：姓名、学号、贡献比例
- 成员 C：姓名、学号、贡献比例
