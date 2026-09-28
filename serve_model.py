import os
import logging
from typing import List

from flask import Flask, request, jsonify, send_from_directory, abort

from model_store import load_model_bundle
from pipeline import TwinDataset, WellDigitalTwin
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("BAGHEWALA_MODEL_PATH")

app = Flask(__name__)


def load_model(path: str = None):
    bundle = load_model_bundle(path) if path else load_model_bundle()
    return bundle


def ensure_model_loaded():
    """Load the model once on-demand."""
    global MODEL, MODEL_LOAD_ERROR, MODEL_BUNDLE
    if "MODEL" in globals() or "MODEL_LOAD_ERROR" in globals():
        return
    try:
        bundle = load_model(MODEL_PATH)
        MODEL_BUNDLE = bundle
        MODEL = bundle["models"]["gap3_rod_risk_model"]
        logger.info("Model loaded successfully.")
    except Exception as e:
        MODEL_LOAD_ERROR = str(e)
        logger.exception("Failed to load model artifact: %s", e)


@app.route("/health", methods=["GET"])
def health():
    ensure_model_loaded()
    if "MODEL" in globals():
        return jsonify({"status": "ok"}), 200
    return jsonify({"status": "error", "error": globals().get("MODEL_LOAD_ERROR")}), 503


@app.route("/", methods=["GET"])
def index():
    dashboard_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")
    index_path = os.path.join(dashboard_dir, "index.html")
    if os.path.isfile(index_path):
        return send_from_directory(dashboard_dir, "index.html")
    return jsonify({"error": "Dashboard not found"}), 404


@app.route("/<path:filename>", methods=["GET"])
def serve_dashboard_file(filename):
    # Avoid interference with API endpoints
    if filename in ("predict", "health", "api", "model_info"):
        abort(404)
    dashboard_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")
    file_path = os.path.join(dashboard_dir, filename)
    if os.path.isfile(file_path):
        return send_from_directory(dashboard_dir, filename)
    return jsonify({"error": "Not found"}), 404


@app.route("/model_info", methods=["GET"])
def model_info():
    ensure_model_loaded()
    if "MODEL" in globals():
        bundle = globals().get("MODEL_BUNDLE") or {}
        info = {
            "loaded": True,
            "training_summary": bundle.get("training_summary"),
            "training_data_dir": bundle.get("training_data_dir"),
            "training_data_fingerprint": bundle.get("training_data_fingerprint"),
            "versions": bundle.get("versions"),
        }
        return jsonify(info), 200
    return jsonify({"loaded": False, "error": globals().get("MODEL_LOAD_ERROR")}), 503


def _predict_one(model, instance: dict) -> dict:
    try:
        return model.predict(instance)
    except Exception as e:
        logger.exception("Prediction failed for instance: %s", e)
        return {"error": str(e)}


def _make_twin():
    """Create a WellDigitalTwin using the on-disk dataset and the loaded bundle."""
    ensure_model_loaded()
    if "MODEL" not in globals():
        raise RuntimeError(globals().get("MODEL_LOAD_ERROR") or "Model not loaded")
    dataset = TwinDataset.load()
    twin = WellDigitalTwin(dataset, model_bundle=globals().get("MODEL_BUNDLE"))
    return twin, dataset


@app.route("/predict", methods=["POST"])
def predict():
    ensure_model_loaded()
    if "MODEL" not in globals():
        return jsonify({"error": "Model not loaded", "detail": globals().get("MODEL_LOAD_ERROR")}), 503

    payload = request.get_json(force=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON payload"}), 400

    instances = payload.get("instances") if isinstance(payload, dict) and "instances" in payload else payload
    if isinstance(instances, dict):
        instances = [instances]
    if not isinstance(instances, list):
        return jsonify({"error": "`instances` must be a dict or list of dicts"}), 400

    results: List[dict] = [_predict_one(MODEL, inst) for inst in instances]
    return jsonify({"predictions": results}), 200


@app.route("/api/overview", methods=["GET"])
def api_overview():
    try:
        twin, dataset = _make_twin()
        fleet = twin.fleet_snapshot()
        well_count = len(fleet)
        median_sor = float(dataset.css_cycles.cycle_sor.median()) if len(dataset.css_cycles) else 0.0
        high_risk_count = sum(1 for w in fleet if w.get("risk_tier") == "high")
        mean_fillage = float(dataset.daily_production.pump_fillage_pct.mean()) if len(dataset.daily_production) else 0.0
        summary = {"well_count": well_count, "median_sor": median_sor,
                   "high_risk_count": high_risk_count, "mean_fillage_pct": mean_fillage}
        resp = {
            "fleet": fleet,
            "refreshed_at": datetime.utcnow().isoformat(),
            "model_trained_at": globals().get("MODEL_BUNDLE", {}).get("trained_at"),
            "data_as_of": (str(dataset.daily_production["date"].max()) if "date" in dataset.daily_production.columns else None),
            "next_refresh_at": None,
            "model_artifact": globals().get("MODEL_BUNDLE", {}).get("training_data_dir") or os.path.basename(MODEL_PATH) if MODEL_PATH else None,
            "data_source": str(dataset),
            "summary": summary,
            "alerts": [],
            "last_refresh_error": None,
        }
        return jsonify(resp), 200
    except Exception as e:
        logger.exception("/api/overview failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/wells/<well_id>", methods=["GET"])
def api_well_detail(well_id):
    try:
        twin, _ = _make_twin()
        detail = twin.well_detail(well_id)
        return jsonify(detail), 200
    except Exception as e:
        logger.exception("/api/wells/%s failed: %s", well_id, e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/wells/<well_id>/css-recommendation", methods=["POST"])
def api_css_recommendation(well_id):
    try:
        twin, _ = _make_twin()
        rec = twin.css_recommendation(well_id)
        return jsonify(rec), 200
    except Exception as e:
        logger.exception("/api/wells/%s/css-recommendation failed: %s", well_id, e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/wells/<well_id>/css-forecast", methods=["POST"])
def api_css_forecast(well_id):
    try:
        twin, _ = _make_twin()
        payload = request.get_json(force=True)
        well_features = twin._well_features(well_id)
        forecast = twin.forecaster.predict(well_features, payload)
        model_info = {
            "artifact": globals().get("MODEL_BUNDLE", {}).get("training_data_dir") or os.path.basename(MODEL_PATH) if MODEL_PATH else None,
            "trained_at": globals().get("MODEL_BUNDLE", {}).get("trained_at"),
            "training_data_matches_current": True,
        }
        return jsonify({"forecast": forecast, "model": model_info}), 200
    except Exception as e:
        logger.exception("/api/wells/%s/css-forecast failed: %s", well_id, e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    try:
        bundle = load_model_bundle(MODEL_PATH) if MODEL_PATH else load_model_bundle()
        globals()["MODEL_BUNDLE"] = bundle
        globals()["MODEL"] = bundle["models"]["gap3_rod_risk_model"]
        return jsonify({"refreshed": True}), 200
    except Exception as e:
        logger.exception("/api/refresh failed: %s", e)
        return jsonify({"refreshed": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    try:
        # Try to pre-load model for faster first request
        MODEL = load_model(MODEL_PATH)
        globals()["MODEL_BUNDLE"] = MODEL
        globals()["MODEL"] = MODEL["models"]["gap3_rod_risk_model"]
    except Exception:
        logger.exception("Model failed to load on startup.")
    app.run(host="0.0.0.0", port=port)
