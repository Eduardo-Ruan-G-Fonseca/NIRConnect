import os
import sys
import json
from fastapi.testclient import TestClient
from typer.testing import CliRunner

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from backend.main import app, METRICS_FILE
from backend.cli import app as cli_app

client = TestClient(app)
runner = CliRunner()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_metrics_upload_and_dashboard(tmp_path):
    metrics = {"R2": 0.9, "RMSE": 0.1, "Accuracy": 0.8}
    resp = client.post("/metrics/upload", json=metrics)
    assert resp.status_code == 200
    assert os.path.exists(METRICS_FILE)

    get_resp = client.get("/metrics")
    assert get_resp.status_code == 200
    assert get_resp.json() == metrics

    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", "test.log")
    with open(log_file, "w") as f:
        f.write("INFO something\nERROR oops\n")

    data_resp = client.get("/dashboard/data")
    assert data_resp.status_code == 200
    data = data_resp.json()
    assert "logs" in data
    assert "model_metrics" in data


def test_cli_report_and_dashboard(tmp_path):
    pdf_path = tmp_path / "report.pdf"
    res = runner.invoke(
        cli_app,
        [
            "report",
            json.dumps({"R2": 1.0, "RMSE": 0.1, "Accuracy": 0.95}),
            "--output",
            str(pdf_path),
        ],
    )
    assert res.exit_code == 0
    assert pdf_path.exists()

    html_path = tmp_path / "dash.html"
    res = runner.invoke(cli_app, ["dashboard", "--output", str(html_path)])
    assert res.exit_code == 0
    assert html_path.exists()


def test_report_endpoint(tmp_path):
    payload = {
        "metrics": {"R2": 0.9},
        "validation_used": "KFold",
        "n_splits_effective": 5,
        "range_used": [400, 700],
        "best": {"preprocess": "none", "n_components": 1, "val_metrics": {"R2CV": 0.9, "RMSECV": 0.1}},
        "curves": [],
    }
    resp = client.post("/report", json=payload)
    assert resp.status_code == 200
    path = resp.json()["path"]
    resp2 = client.get("/report/download", params={"path": path})
    assert resp2.status_code == 200


def test_process_file(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [2, 4, 6, 8], "target": [1, 0, 1, 0]})
    path = tmp_path / "data.xlsx"
    df.to_excel(path, index=False)
    with open(path, "rb") as fh:
        resp = client.post(
            "/process",
            files={"file": ("data.xlsx", fh.read(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            data={"target": "target", "n_components": 2, "classification": "true", "decision_mode": "argmax"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "metrics" in data
    assert "vip" in data
    assert "top_vips" in data and isinstance(data["top_vips"], list)


def test_columns_return_mean(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"1100": [1.0, 2.0], "1200": [2.0, 4.0], "target": [1, 0]})
    path = tmp_path / "spectra.csv"
    df.to_csv(path, index=False)
    with open(path, "rb") as fh:
        resp = client.post(
            "/columns",
            files={"file": ("spectra.csv", fh.read(), "text/csv")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "mean_spectra" in data
    ms = data["mean_spectra"]
    assert ms["wavelengths"] == [1100.0, 1200.0]
    assert len(ms["values"]) == 2
    assert "spectra_matrix" in data
    sm = data["spectra_matrix"]
    assert sm["wavelengths"] == [1100.0, 1200.0]
    assert len(sm["values"]) == 2


def test_columns_ignore_textual(tmp_path):
    import pandas as pd
    df = pd.DataFrame(
        {
            "NOM. POPULAR": ["a", "b"],
            "1100": [1.0, 2.0],
            "1450.5": [2.5, 3.5],
            "Valor": [10, 20],
        }
    )
    path = tmp_path / "mix.csv"
    df.to_csv(path, index=False)
    with open(path, "rb") as fh:
        resp = client.post(
            "/columns",
            files={"file": ("mix.csv", fh.read(), "text/csv")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["spectra"]) == {"1100", "1450.5"}
    ms = data["mean_spectra"]
    assert ms["wavelengths"] == [1100.0, 1450.5]
    sm = data["spectra_matrix"]
    assert sm["wavelengths"] == [1100.0, 1450.5]
    assert len(sm["values"]) == 2


def test_columns_blank_lines_and_excel_first_sheet(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"A": [1, 2], "B": [2, 3], "target": [0, 1]})
    csv_path = tmp_path / "blank.csv"
    df.to_csv(csv_path, index=False)
    text = csv_path.read_text()
    csv_path.write_text("\n" + text)
    with open(csv_path, "rb") as fh:
        resp = client.post("/columns", files={"file": ("blank.csv", fh.read(), "text/csv")})
    assert resp.status_code == 200
    cols = [c["name"] for c in resp.json()["columns"]]
    assert "A" in cols and "B" in cols

    x1 = pd.DataFrame({"X": [1], "target": [0]})
    x2 = pd.DataFrame({"Y": [2], "target": [1]})
    xl_path = tmp_path / "multi.xlsx"
    with pd.ExcelWriter(xl_path) as writer:
        x1.to_excel(writer, index=False, sheet_name="S1")
        x2.to_excel(writer, index=False, sheet_name="S2")
    with open(xl_path, "rb") as fh:
        resp = client.post(
            "/columns",
            files={"file": ("multi.xlsx", fh.read(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
    assert resp.status_code == 200
    cols = [c["name"] for c in resp.json()["columns"]]
    assert "X" in cols and "Y" not in cols


def test_columns_decimal_comma_and_invalid(tmp_path):
    import pandas as pd
    df = pd.DataFrame({
        "908,1": [0.1, 0.2],
        "920,4": [0.3, 0.4],
        "sample": [1, 2],
        "1100a": [5, 6],
    })
    xl_path = tmp_path / "comma.xlsx"
    df.to_excel(xl_path, index=False)
    with open(xl_path, "rb") as fh:
        resp = client.post(
            "/columns",
            files={
                "file": (
                    "comma.xlsx",
                    fh.read(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["spectra"]) == {"908,1", "920,4"}
    assert set(data["targets"]) == {"sample", "1100a"}
    assert any("1100a" in w for w in data.get("warnings", []))


def test_analisar_ranges_and_history(tmp_path):
    import pandas as pd, json as js
    cols = {str(1100 + i*10): [i+j for j in range(4)] for i in range(4)}
    df = pd.DataFrame(cols)
    df["target"] = [1.0, 2.0, 3.0, 4.0]
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    with open(path, "rb") as fh:
        resp = client.post(
            "/analisar",
            files={"file": ("d.csv", fh.read(), "text/csv")},
            data={
                "target": "target",
                "n_components": 2,
                "spectral_ranges": "1100-1120",
                "preprocess": "zscore",
                "n_bootstrap": 0,
                "classification": "false",
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "y_real" in data and "y_pred" in data
    assert "top_vips" in data
    assert "resumo_interpretativo" in data
    history_file = os.path.join("models", "history.json")
    assert os.path.exists(history_file)
    with open(history_file) as f:
        hist = js.load(f)
    assert len(hist) >= 1
    assert hist[-1]["preprocessing"] == ["Z-score"]
    assert "top_vips" in hist[-1]
    assert hist[-1]["tipo_analise"] == "PLS-R"


def test_analisar_classification(tmp_path):
    import pandas as pd, json as js
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [1, 3, 2, 4], "target": [0, 1, 0, 1]})
    path = tmp_path / "c.csv"
    df.to_csv(path, index=False)
    with open(path, "rb") as fh:
        resp = client.post(
            "/analisar",
            files={"file": ("c.csv", fh.read(), "text/csv")},
            data={
                "target": "target",
                "n_components": 2,
                "spectral_ranges": "1100-1120",
                "preprocess": "",
                "n_bootstrap": 0,
                "classification": "true",
                "threshold": 0.4,
                "decision_mode": "argmax",
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["analysis_type"] == "PLS-DA"
    assert "Accuracy" in data["metrics"]
    assert "resumo_interpretativo" in data
    history_file = os.path.join("models", "history.json")
    with open(history_file) as f:
        hist = js.load(f)
    assert hist[-1]["tipo_analise"] == "PLS-DA"


def test_analisar_multiclass(tmp_path):
    import pandas as pd
    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5, 6],
        "B": [1, 3, 2, 4, 5, 6],
        "target": ["x", "y", "z", "x", "y", "z"],
    })
    path = tmp_path / "m.csv"
    df.to_csv(path, index=False)
    with open(path, "rb") as fh:
        resp = client.post(
            "/analisar",
            files={"file": ("m.csv", fh.read(), "text/csv")},
            data={
                "target": "target",
                "n_components": 2,
                "spectral_ranges": "1100-1120",
                "preprocess": "",
                "n_bootstrap": 0,
                "classification": "true",
                "decision_mode": "argmax",
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["analysis_type"] == "PLS-DA"
    assert len(data["metrics"]["confusion_matrix"]) == 3


def test_analisar_with_custom_cv(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [1, 3, 2, 4], "target": [1, 2, 3, 4]})
    path = tmp_path / "cv.csv"
    df.to_csv(path, index=False)
    with open(path, "rb") as fh:
        resp = client.post(
            "/analisar",
            files={"file": ("cv.csv", fh.read(), "text/csv")},
            data={
                "target": "target",
                "n_components": 2,
                "spectral_ranges": "1100-1200",
                "preprocess": "",
                "n_bootstrap": 0,
                "classification": "false",
                "validation_method": "Holdout",
                "validation_params": json.dumps({"test_size": 0.3}),
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "metrics" in data
