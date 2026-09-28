import os
import logging
from typing import List

from flask import Flask, request, jsonify

from model_store import load_model_bundle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("BAGHEWALA_MODEL_PATH")

app = Flask(__name__)


def load_model(path: str = None):
    bundle = load_model_bundle(path) if path else load_model_bundle()
    return bundle["models"]["gap3_rod_risk_model"]


def ensure_model_loaded():
    """Load the model once on-demand. Avoids using removed Flask hooks."""
    global MODEL, MODEL_LOAD_ERROR
    if "MODEL" in globals() or "MODEL_LOAD_ERROR" in globals():
        return
    try:
        MODEL = load_model(MODEL_PATH)
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
