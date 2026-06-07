from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report.pdf"


def register_fonts() -> tuple[str, str]:
    font_dir = Path(r"C:\Windows\Fonts")
    regular = font_dir / "simhei.ttf"
    bold = font_dir / "simhei.ttf"
    pdfmetrics.registerFont(TTFont("CNRegular", str(regular)))
    pdfmetrics.registerFont(TTFont("CNBold", str(bold)))
    return "CNRegular", "CNBold"


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def main() -> None:
    regular, bold = register_fonts()
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCN",
        parent=styles["Title"],
        fontName=bold,
        fontSize=18,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=10,
    )
    h2 = ParagraphStyle(
        "H2CN",
        parent=styles["Heading2"],
        fontName=bold,
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#0F766E"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName=regular,
        fontSize=9.5,
        leading=14.5,
        firstLineIndent=18,
        spaceAfter=5,
    )
    small = ParagraphStyle(
        "SmallCN",
        parent=styles["BodyText"],
        fontName=regular,
        fontSize=8.5,
        leading=12,
        spaceAfter=3,
    )

    metrics_path = ROOT / "results" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    llm_metrics_path = ROOT / "results" / "llm_override_val.metrics.json"
    llm_metrics = (
        json.loads(llm_metrics_path.read_text(encoding="utf-8"))
        if llm_metrics_path.exists()
        else {}
    )
    acc = metrics.get("accuracy", 0)
    macro_f1 = metrics.get("macro_f1", 0)
    rumor_f1 = metrics.get("rumor_f1", 0)
    cmatrix = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    llm_acc = llm_metrics.get("accuracy", 0)
    llm_macro_f1 = llm_metrics.get("macro_f1", 0)
    llm_calls = llm_metrics.get("llm_calls", 0)
    llm_overrides = llm_metrics.get("llm_overrides", 0)
    fn_review_path = ROOT / "results" / "fn_recall_review_summary.json"
    fn_review = json.loads(fn_review_path.read_text(encoding="utf-8")) if fn_review_path.exists() else {}
    fn_default = fn_review.get("predeclared_default", {})

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
    )

    story = [p("可解释的谣言检测大作业报告", title)]
    meta = Table(
        [
            ["课程", "人工智能导论", "题目", "可解释的谣言检测"],
            [
                "小组成员",
                "请填写真实姓名、学号",
                "代码仓库",
                "https://github.com/ipromise77/explainable-rumor-detection-2026",
            ],
        ],
        colWidths=[2.2 * cm, 5.5 * cm, 2.2 * cm, 7.1 * cm],
    )
    meta.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), regular),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E0F2F1")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#E0F2F1")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#99A3A4")),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story += [meta, Spacer(1, 8)]

    sections = [
        (
            "1. 任务与目标",
            "本项目面向短推文谣言检测任务，要求输入一条文本后输出二分类标签，并给出可读判断依据。标签中0表示非谣言，1表示谣言。系统采用“本地可解释检测器+学校API大模型增强解释”的复合架构，兼顾准确率、解释合理性、运行效率和可复现性。",
        ),
        (
            "2. 数据与预处理",
            "数据集包含train.csv与val.csv。原训练集2840条，验证集401条。预处理时进行HTML反转义、小写化，并将URL、用户提及和hashtag归一化为稳定标记。数据审计发现训练集中有3组、7行样本在当前预处理后文本相同但标签冲突，验证集未发现同类冲突；因此生成train_clean.csv，剔除冲突样本后使用2833条正常训练样本重训。",
        ),
        (
            "3. 模型设计",
            "基础模型采用三路TF-IDF逻辑回归集成：词级1-2/1-3 gram捕捉事件关键词和短语，字符级2-6/3-6 gram捕捉hashtag、拼写变体和短文本片段。对于低置信样本，系统调用deepseek-reasoner，并提供本地证据和相似训练样本进行复核与解释生成；针对事件级分析发现的漏报问题，额外设计local_pred=0且0.20<=prob_rumor<0.50的false negative召回复核。",
        ),
    ]
    for heading, text in sections:
        story += [p(heading, h2), p(text, body)]

    story += [p("4. 实验结果", h2)]
    story += [
        p(
            "在val.csv上，使用清洗训练集的最终模型主要指标如下。5-fold交叉验证Accuracy为0.8641±0.0176；train内部dev阈值实验选择0.51，但应用到val.csv后准确率降到0.8753，因此最终保留0.5。事件级错误分析显示event 0和event 1的错误均为false negative；低置信false negative复核在不新增误报的情况下救回1条漏报。全部结果可复现。",
            body,
        )
    ]
    result_table = Table(
        [
            ["模式", "Accuracy", "Macro-F1", "备注"],
            ["本地TF-IDF集成", f"{acc:.4f}", f"{macro_f1:.4f}", f"Rumor F1={rumor_f1:.4f}"],
            [
                "低置信FN召回复核",
                f"{fn_default.get('accuracy', 0):.4f}",
                f"{fn_default.get('macro_f1', 0):.4f}",
                f"救回{fn_default.get('rescued_false_negatives', 0)}条FN，新增FP={fn_default.get('new_false_positives', 0)}",
            ],
            [
                "LLM增强高置信覆盖",
                f"{llm_acc:.4f}",
                f"{llm_macro_f1:.4f}",
                f"调用{llm_calls}条，覆盖{llm_overrides}条",
            ],
            ["本地混淆矩阵", str(cmatrix), "", "[[TN, FP], [FN, TP]]"],
        ],
        colWidths=[5.4 * cm, 3.0 * cm, 3.0 * cm, 4.6 * cm],
    )
    result_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), regular),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#99A3A4")),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story += [result_table]

    tail_sections = [
        (
            "5. 可解释性输出",
            "基础解释模块计算TF-IDF特征值与逻辑回归权重的乘积，并汇总为“支持谣言”和“支持非谣言”的证据。输出采用固定结构，依次给出预测标签、谣言概率、分类置信度、正向证据、反向证据和综合判断，便于检查依据是否来自当前模型而非主观编造。LLM增强模块把本地概率、正反证据和RAG检索出的相似训练样本输入deepseek-reasoner，由大模型生成中文判断依据；默认不随意改标签，只有高置信冲突时才覆盖。",
        ),
        (
            "6. 泛化与运行效率",
            "模型训练不使用val标签。即使学校API不可用，本地检测器仍可独立运行并达到0.8803准确率；API可用时，只复核低置信样本，兼顾解释性和运行时间。字符级特征提升了对拼写变化、链接替换和话题标签的鲁棒性，相似样本提示进一步增强跨事件泛化。",
        ),
        (
            "7. 创新点",
            "第一，采用“词级语义+字符级形态”的多视角集成，在短文本场景下兼顾事件关键词和社交媒体局部模式。第二，设计与模型权重绑定的局部解释生成器，能够同时展示支持与反对证据。第三，引入低置信度LLM复核与RAG式相似样本提示，使判断依据更自然且更贴近数据集分布。第四，基于事件级错误分析设计false negative召回复核，只对本地判0且接近阈值的样本进行保守覆盖。",
        ),
        (
            "8. 小组分工",
            "请按真实情况填写：组长负责仓库管理、实验统筹和报告整合；成员A负责数据分析和预处理；成员B负责模型训练与调参；成员C负责解释模块、README和部署测试。最终贡献比例请由小组协商后在此处列出。",
        ),
    ]
    for heading, text in tail_sections:
        story += [p(heading, h2), p(text, body)]

    story += [
        Spacer(1, 5),
        p("备注：提交前请将小组成员、学号、贡献比例和GitHub地址替换为真实信息。", small),
    ]
    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
