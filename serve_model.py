import os
import logging
from typing import List

from flask import Flask, request, jsonify, send_from_directory, abort

from model_store import load_model_bundle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("BAGHEWALA_MODEL_PATH")

app = Flask(__name__)


def load_model(path: str = None):
    bundle = load_model_bundle(path) if path else load_model_bundle()
    return bundle


def ensure_model_loaded():
    """Load the model once on-demand. Avoids using removed Flask hooks."""
    global MODEL, MODEL_LOAD_ERROR
    if "MODEL" in globals() or "MODEL_LOAD_ERROR" in globals():
        return
    try:
        bundle = load_model(MODEL_PATH)
        MODEL = bundle["models"]["gap3_rod_risk_model"]
        MODEL_BUNDLE = bundle
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
    # Serve dashboard/index.html if present in the repo's `dashboard/` folder
    dashboard_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")
    index_path = os.path.join(dashboard_dir, "index.html")
    if os.path.isfile(index_path):
        return send_from_directory(dashboard_dir, "index.html")
    return jsonify({"error": "Dashboard not found"}), 404


@app.route("/<path:filename>", methods=["GET"])
def serve_dashboard_file(filename):
    # Avoid interfering with API endpoints like /predict and /health
    if filename in ("predict", "health"):
        abort(404)
    dashboard_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")
    file_path = os.path.join(dashboard_dir, filename)
    if os.path.isfile(file_path):
        return send_from_directory(dashboard_dir, filename)
    return jsonify({"error": "Not found"}), 404


@app.route("/model_info", methods=["GET"])
def model_info():
    """Return model bundle metadata or the load error for debugging."""
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


@app.route("/predict", methods=["POST"])
def predict():
    ensure_model_loaded()
    if "MODEL" not in globals():
        return jsonify({"error": "Model not loaded", "detail": globals().get("MODEL_LOAD_ERROR")}), 503

    payload = request.get_json(force=True)
    # Accept either {"instances": [..]} or a single dict payload
    if payload is None:
        return jsonify({"error": "Invalid JSON payload"}), 400

    instances = payload.get("instances") if isinstance(payload, dict) and "instances" in payload else payload

    # Normalize to list
    if isinstance(instances, dict):
        instances = [instances]
    if not isinstance(instances, list):
        return jsonify({"error": "`instances` must be a dict or list of dicts"}), 400

    results: List[dict] = [_predict_one(MODEL, inst) for inst in instances]
    return jsonify({"predictions": results}), 200


if __name__ == "__main__":
    # For local development only; production should use gunicorn
    port = int(os.getenv("PORT", 8080))
    try:
        MODEL = load_model(MODEL_PATH)
    except Exception:
        logger.exception("Model failed to load on startup.")
    app.run(host="0.0.0.0", port=port)
