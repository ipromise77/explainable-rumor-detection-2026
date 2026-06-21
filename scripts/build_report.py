from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEX_FILE = ROOT / "report.tex"
PDF_FILE = ROOT / "report.pdf"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def build_with_latex() -> bool:
    latexmk = shutil.which("latexmk")
    xelatex = shutil.which("xelatex")

    if latexmk:
        run(
            [
                latexmk,
                "-xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                TEX_FILE.name,
            ]
        )
        return PDF_FILE.exists()

    if xelatex:
        cmd = [xelatex, "-interaction=nonstopmode", "-halt-on-error", TEX_FILE.name]
        run(cmd)
        run(cmd)
        return PDF_FILE.exists()

    return False


def register_cjk_font() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for path in candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont("CJK", str(path)))
            return "CJK"
    return "Helvetica"


def build_fallback_pdf() -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Flowable,
        ListFlowable,
        ListItem,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    font = register_cjk_font()
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "body",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=9.2,
        leading=14,
        wordWrap="CJK",
        spaceAfter=5,
    )
    title = ParagraphStyle(
        "title",
        parent=body,
        alignment=TA_CENTER,
        fontSize=19,
        leading=24,
        spaceAfter=8,
    )
    subtitle = ParagraphStyle(
        "subtitle",
        parent=body,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10,
    )
    h1 = ParagraphStyle(
        "h1",
        parent=body,
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#0F766E"),
        spaceBefore=7,
        spaceAfter=4,
    )
    small = ParagraphStyle("small", parent=body, fontSize=8.2, leading=12)

    class Rule(Flowable):
        def __init__(self, color: str = "#0F766E"):
            super().__init__()
            self.color = colors.HexColor(color)

        def wrap(self, avail_width, avail_height):
            self.width = avail_width
            self.height = 0.12 * cm
            return self.width, self.height

        def draw(self):
            self.canv.setStrokeColor(self.color)
            self.canv.setLineWidth(1)
            self.canv.line(0, 0.05 * cm, self.width, 0.05 * cm)

    class FlowDiagram(Flowable):
        def __init__(self, labels: list[str], caption: str):
            super().__init__()
            self.labels = labels
            self.caption = caption

        def wrap(self, avail_width, avail_height):
            self.width = avail_width
            self.height = 2.1 * cm
            return self.width, self.height

        def draw(self):
            canvas = self.canv
            gap = 0.28 * cm
            box_w = (self.width - gap * (len(self.labels) - 1)) / len(self.labels)
            box_h = 0.78 * cm
            y = 0.72 * cm
            canvas.setFont(font, 7.2)
            for i, label in enumerate(self.labels):
                x = i * (box_w + gap)
                canvas.setFillColor(colors.HexColor("#EEF7F5"))
                canvas.setStrokeColor(colors.HexColor("#0F766E"))
                canvas.roundRect(x, y, box_w, box_h, 4, fill=1, stroke=1)
                canvas.setFillColor(colors.black)
                for j, part in enumerate(label.split("\\n")):
                    canvas.drawCentredString(
                        x + box_w / 2,
                        y + box_h / 2 + (0.12 - 0.22 * j) * cm,
                        part,
                    )
                if i < len(self.labels) - 1:
                    ax1 = x + box_w + 0.04 * cm
                    ax2 = x + box_w + gap - 0.04 * cm
                    ay = y + box_h / 2
                    canvas.setStrokeColor(colors.HexColor("#475569"))
                    canvas.line(ax1, ay, ax2, ay)
                    canvas.line(ax2, ay, ax2 - 0.10 * cm, ay + 0.05 * cm)
                    canvas.line(ax2, ay, ax2 - 0.10 * cm, ay - 0.05 * cm)
            canvas.setFont(font, 8)
            canvas.setFillColor(colors.HexColor("#475569"))
            canvas.drawCentredString(self.width / 2, 0.18 * cm, self.caption)

    def p(text: str, style=body) -> Paragraph:
        return Paragraph(text, style)

    def bullet(items: list[str]) -> ListFlowable:
        return ListFlowable(
            [ListItem(p(item, body), leftIndent=10) for item in items],
            bulletType="bullet",
            start="circle",
            leftIndent=12,
        )

    def table(rows: list[list[str]], widths: list[float]) -> Table:
        data = [[p(cell, small) for cell in row] for row in rows]
        t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6F4F1")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F3F3A")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return t

    doc = SimpleDocTemplate(
        str(PDF_FILE),
        pagesize=A4,
        leftMargin=1.85 * cm,
        rightMargin=1.85 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.7 * cm,
        title="可解释的谣言检测",
    )

    story = [
        p("可解释的谣言检测", title),
        p("人工智能导论大作业 2026 | 小组成员：待填写姓名、学号、贡献比例", subtitle),
        Rule(),
        p("<b>摘要。</b>本项目实现英文推文谣言检测系统，输入文本后输出 0/1 标签和中文判断依据。默认主模型为三路 TF-IDF + Logistic Regression 集成，使用 train_clean.csv 训练，不依赖学校 API 或本地缓存，在 val.csv 上复现 Accuracy 0.8803。备选增强模型 FinalRumourDetectClass 使用可复现的分级话术信号，Accuracy 达到 0.8928。项目同时保留数据清理、事件级错误分析、阈值实验、5-fold 交叉验证、LLM 复核和失败实验记录。"),
        p("<b>关键词：</b>谣言检测；可解释机器学习；TF-IDF；逻辑回归；RAG；大语言模型复核"),
        p("1. 任务与合规性", h1),
        p("课程要求模型对推文输出 0/1 标签，并给出一段判断依据。GitHub 需包含 README、report.pdf、代码和支持文件。本项目默认检测类为 RumourDetectClass，提供 classify(text)、explain(text) 和 predict(text)。默认复现路径只依赖仓库中的数据、模型和 Python 依赖；学校 API 的 deepseek-reasoner 仅作为可选解释增强与低置信复核实验。"),
        p("2. 数据质量与预处理", h1),
        p("原始训练集 2840 条，验证集 401 条。预处理包括 HTML 反转义、小写化、URLTOKEN、USERTOKEN 和 hashtag 展开。数据审计发现训练集中存在 3 组、7 行预处理后文本相同但标签冲突的异常样本；验证集无同类冲突。因此保留原始 train.csv，额外生成 train_clean.csv 训练。"),
        p("3. 模型架构", h1),
        FlowDiagram(
            ["原始推文", "清洗归一化", "三路 TF-IDF", "逻辑回归\\n概率集成", "标签与解释"],
            "图 1：默认主模型流程",
        ),
        table(
            [
                ["阶段", "输入/处理", "输出"],
                ["清洗", "HTML 反转义、URL/用户/hashtag 归一化", "稳定文本表示"],
                ["特征", "词级 1-2/1-3 gram + 字符级 2-6/3-6 gram", "三路 TF-IDF 向量"],
                ["分类", "三组 Logistic Regression 概率平均", "谣言概率与 0/1 标签"],
                ["解释", "TF-IDF 特征值与权重乘积", "支持谣言/非谣言的局部证据"],
                ["增强", "低概率非谣言样本检查分级话术信号", "备选高召回预测"],
            ],
            [2.2 * cm, 9.5 * cm, 4.2 * cm],
        ),
        p("默认模型训练三组互补管线，解释模块计算当前文本中每个 TF-IDF 特征值与模型权重的乘积。解释不引入外部事实，只反映模型在当前样本上的局部判定依据。"),
        p("4. 实验与结果", h1),
        table(
            [
                ["方案", "Accuracy", "Macro-F1", "Rumor F1", "说明"],
                ["RumourDetectClass", "0.8803", "0.8757", "0.8519", "默认主模型，TN/FP/FN/TP=215/11/37/138"],
                ["低置信 FN 复核", "0.8828", "0.8784", "0.8554", "复核 79 条，覆盖 1 条，无新增 FP"],
                ["LLM 高置信覆盖", "0.8828", "0.8787", "0.8563", "调用 70 条，覆盖 3 条，仅作增强实验"],
                ["FinalRumourDetectClass", "0.8928", "0.8892", "0.8693", "默认不读 LLM 缓存，TN/FP/FN/TP=215/11/32/143"],
            ],
            [4.1 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 6.0 * cm],
        ),
        p("5-fold 交叉验证 Accuracy 为 0.8641±0.0176。train 内部 dev 阈值实验选择 0.51，但应用到 val.csv 后 Accuracy 降至 0.8753，因此最终保留 0.5。事件级分析发现 event 0 和 event 1 的错误主要是 false negative，说明模型偏保守。"),
        p("5. 增强方案与失败实验", h1),
        FlowDiagram(
            ["本地概率 p", "判 0 且\\np≤0.25", "检查话术\\n信号", "强信号或\\np+bias≥0.50", "覆盖为谣言"],
            "图 2：备选增强模型的分级信号流程",
        ),
        p("FinalRumourDetectClass 只在本地模型判为非谣言且 prob_rumor≤0.25 时检查分级话术信号。强信号直接覆盖为谣言，中等信号仅在 prob_rumor+bias≥0.50 时覆盖。远程分支曾记录 0.9027 Accuracy，但依赖验证集后处理脚本与本地缓存，因此仅作为探索记录。稠密语义特征、event 特征、L1 正则化、直接 LLM 接管等尝试未稳定提升效果，已在 docs/ 中作为失败/风险记录保留。"),
        p("6. 可解释性与演示", h1),
        p("默认输出固定包含标签、谣言概率、置信度、支持谣言证据、支持非谣言证据和综合判断。LLM 增强模块采用相似样本提示，将本地证据、近邻标签和输入推文发送给学校 API；高置信覆盖门槛为 0.85。运行 python scripts/run_demo_server.py 可打开本地前端页面，展示标签、概率、来源和中文判断依据。"),
        p("7. 复现方式与分工", h1),
        p("核心命令：pip install -r requirements.txt；python -m src.train --with-explanations；python -m src.evaluate --show-examples 0；python scripts/run_final_detector_evaluation.py；python scripts/run_demo_server.py。小组姓名、学号和贡献比例提交前填写。"),
        p("8. 结论", h1),
        p("本项目以可复现的 TF-IDF 集成模型作为正式主线，在 val.csv 上达到 0.8803 Accuracy，并提供基于局部特征贡献的解释。备选增强模型在不依赖 LLM 缓存的前提下达到 0.8928 Accuracy，展示了由错误分析驱动的召回改进。报告明确区分默认指标、增强实验和探索性失败结果。"),
        p("参考文献", h1),
        bullet(
            [
                "Shu et al. Fake News Detection on Social Media: A Data Mining Perspective. ACM SIGKDD Explorations, 2017.",
                "Salton and Buckley. Term-weighting approaches in automatic text retrieval. Information Processing & Management, 1988.",
                "Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR, 2011.",
                "Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS, 2020.",
            ]
        ),
        p("附录：核心接口", h1),
        p("from src.rumor_detector import RumourDetectClass"),
        p("from src.final_detector import FinalRumourDetectClass"),
        p("main_detector = RumourDetectClass(); label = main_detector.classify('input tweet text'); reason = main_detector.explain('input tweet text')"),
        p("final_detector = FinalRumourDetectClass(); result = final_detector.predict('input tweet text')"),
    ]

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawRightString(A4[0] - 1.85 * cm, 1.0 * cm, str(canvas.getPageNumber()))
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def main() -> int:
    if not TEX_FILE.exists():
        print(f"Missing LaTeX source: {TEX_FILE}", file=sys.stderr)
        return 1

    if build_with_latex():
        print(f"Wrote {PDF_FILE}")
        return 0

    print("No LaTeX compiler found; writing ReportLab fallback PDF from the same report content.")
    build_fallback_pdf()
    print(f"Wrote {PDF_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
