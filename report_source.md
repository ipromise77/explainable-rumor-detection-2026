# 报告源说明

正式提交报告源文件为 `report.tex`，最终 PDF 为 `report.pdf`。

本文件不再作为正文草稿，避免与 LaTeX 报告产生口径冲突。提交前请检查：

- `report.tex` 已填写真实姓名、学号和贡献比例。
- 已安装 TeX Live 或 MiKTeX，并运行 `python scripts/build_report.py` 生成最新 `report.pdf`。
- 默认主模型 `RumourDetectClass` 指标为 Accuracy `0.8803`。
- 备选增强模型 `FinalRumourDetectClass` 默认不依赖 ignored LLM cache，指标为 Accuracy `0.8928`。
- 远程历史分支 `feat/rule-signal-system` 的 `0.9027` 仅作为探索记录，不作为默认主模型指标。
