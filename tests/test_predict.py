#!/usr/bin/env python3
import sys
import os
import json
from pathlib import Path

# ensure repo root is on sys.path so local modules can be imported when
# running this script from the workspace root (e.g. `python tests/test_predict.py`).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_store import load_model_bundle


def main():
    model_path = Path(__file__).resolve().parents[1] / "model_artifacts" / "baghewala_models.pkl"
    try:
        bundle = load_model_bundle(model_path)
    except Exception as e:
        print("Failed to load model bundle:", e, file=sys.stderr)
        sys.exit(2)

    model = bundle["models"]["gap3_rod_risk_model"]
    sample = {
        "rod_age_days": 100,
        "cumulative_rod_float_days": 2,
        "avg_impact_load_proxy_last_30d": 0.5,
        "rod_material": "steel",
        "depth_m": 1500,
    }
    pred = model.predict(sample)
    print(json.dumps({"sample": sample, "prediction": pred}, indent=2))


if __name__ == "__main__":
    main()
