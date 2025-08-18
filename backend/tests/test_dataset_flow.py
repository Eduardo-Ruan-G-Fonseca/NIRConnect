import pandas as pd
from fastapi.testclient import TestClient

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import app


client = TestClient(app)


def test_upload_preprocess_and_train(tmp_path):
    df = pd.DataFrame({"1100": [1, 2, 3, 4], "1200": [2, 4, 6, 8], "target": [0, 1, 0, 1]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    with open(path, "rb") as fh:
        resp = client.post("/columns", files={"file": ("data.csv", fh.read(), "text/csv")})
    assert resp.status_code == 200
    dataset_id = resp.json()["dataset_id"]
    assert dataset_id

    resp = client.post(
        "/preprocess",
        json={"dataset_id": dataset_id, "target": "target", "spectral_ranges": "1100-1200"},
    )
    assert resp.status_code == 200
    cols = resp.json()["columns"]
    assert "1100" in cols and "1200" in cols

    train_payload = {
        "dataset_id": dataset_id,
        "target": "target",
        "analysis_mode": "PLS-DA",
        "n_components": 2,
        "preprocess": [],
        "spectral_ranges": "1100-1200",
        "validation_method": "LOO",
        "decision_mode": "argmax",
    }
    resp = client.post("/train", json=train_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "model_id" in data

