import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backend.core.report_pdf import PDFReport


def test_pdf_creation(tmp_path):
    metrics = {"R2": 0.95, "RMSE": 0.1}
    output = tmp_path / "report.pdf"
    pdf = PDFReport()
    pdf.add_metrics(metrics)
    pdf.output(str(output))
    assert output.exists()
    assert output.stat().st_size > 0


def test_pdf_classification(tmp_path):
    metrics = {
        "Accuracy": 0.9,
        "F1_macro": 0.9,
        "F1_micro": 0.9,
        "Kappa": 0.9,
        "confusion_matrix": [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
    }
    params = {
        "analysis_type": "PLS-DA",
        "decision_mode": "argmax",
        "class_mapping": {0: "A", 1: "B", 2: "C"},
    }
    output = tmp_path / "cls_report.pdf"
    pdf = PDFReport()
    pdf.add_metrics(metrics, params=params)
    pdf.output(str(output))
    assert output.exists()
    assert output.stat().st_size > 0
