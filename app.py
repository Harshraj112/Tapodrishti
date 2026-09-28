"""HTTP backend for the Baghewala Well-to-Surface Digital Twin dashboard."""

from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
from threading import Event, Lock, Thread
from time import monotonic
from urllib.parse import unquote, urlsplit

import numpy as np
import pandas as pd

from model_store import DEFAULT_MODEL_PATH, dataset_fingerprint, load_model_bundle
from pipeline import BOUNDS, TwinDataset, WellDigitalTwin


ROOT = Path(__file__).resolve().parent
DASHBOARD_DIR = ROOT / "dashboard"


def resolve_data_dir() -> Path:
    configured = os.getenv("BAGHEWALA_DATA_DIR")
    if configured:
        candidate = Path(configured)
        return candidate if candidate.is_absolute() else ROOT / candidate
    generated = ROOT / "baghewala_dataset"
    return generated if generated.is_dir() else ROOT / "baghewala_sample_dataset"


DATA_DIR = resolve_data_dir()
if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")
MODEL_PATH = Path(os.getenv("BAGHEWALA_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = ROOT / MODEL_PATH

REFRESH_INTERVAL_SECONDS = max(60, int(os.getenv("BAGHEWALA_REFRESH_SECONDS", "900")))


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    return str(value)


class TwinService:
    def __init__(self, data_dir: Path, model_path: Path, refresh_interval_seconds: int):
        self.data_dir = data_dir
        self.model_path = model_path
        self.refresh_interval_seconds = refresh_interval_seconds
        self._state_lock = Lock()
        self._refresh_lock = Lock()
        self._stop_event = Event()
        self._worker = None
        dataset = TwinDataset.load(str(data_dir))
        bundle = load_model_bundle(model_path)
        twin = WellDigitalTwin(dataset, model_bundle=bundle)
        self._state = self._build_state(dataset, twin, bundle)
        self._next_refresh_monotonic = monotonic() + self.refresh_interval_seconds

    @staticmethod
    def _alerts_for_change(previous_fleet, current_fleet, detected_at):
        previous_by_well = {row["well_id"]: row for row in previous_fleet}
        risk_rank = {"low": 0, "medium": 1, "high": 2}
        alerts = []
        for current in current_fleet:
            old = previous_by_well.get(current["well_id"])
            if old is None:
                continue
            old_risk = old.get("risk_tier", "low")
            new_risk = current.get("risk_tier", "low")
            if old_risk != new_risk:
                increased = risk_rank.get(new_risk, 0) > risk_rank.get(old_risk, 0)
                alerts.append({
                    "well_id": current["well_id"],
                    "type": "risk_tier_changed",
                    "severity": new_risk if increased else "info",
                    "detected_at": detected_at,
                    "message": f"Rod risk changed from {old_risk} to {new_risk}.",
                    "previous": old_risk,
                    "current": new_risk,
                })

            old_fault = old.get("card_label", "unknown")
            new_fault = current.get("card_label", "unknown")
            if old_fault != new_fault and new_fault not in {"normal", "unknown", "n/a"}:
                alerts.append({
                    "well_id": current["well_id"],
                    "type": "fault_classification_changed",
                    "severity": "high" if new_fault in {"parted_rod", "rod_float"} else "medium",
                    "detected_at": detected_at,
                    "message": f"Card classification changed from {old_fault} to {new_fault}.",
                    "previous": old_fault,
                    "current": new_fault,
                })
        return alerts

    def _build_state(self, dataset, twin, bundle, previous=None):
        refreshed_at = datetime.now().astimezone().isoformat(timespec="seconds")
        fleet = twin.fleet_snapshot()
        sor_values = [row["latest_sor"] for row in fleet if row.get("latest_sor") is not None]
        fillage_values = [row["latest_fillage_pct"] for row in fleet
                          if row.get("latest_fillage_pct") is not None]
        data_dates = pd.to_datetime(dataset.daily_production["date"], errors="coerce")
        data_as_of = data_dates.max()
        data_as_of = data_as_of.isoformat() if pd.notna(data_as_of) else None
        changes = self._alerts_for_change(previous["fleet"], fleet, refreshed_at) if previous else []
        alerts = ([*previous.get("alerts", []), *changes][-50:] if previous else [])
        return {
            "dataset": dataset,
            "twin": twin,
            "fleet": json_safe(fleet),
            "summary": {
                "well_count": len(fleet),
                "median_sor": float(np.median(sor_values)) if sor_values else None,
                "mean_fillage_pct": float(np.mean(fillage_values)) if fillage_values else None,
                "high_risk_count": sum(row["risk_tier"] == "high" for row in fleet),
                "rod_float_count": sum(row["card_label"] == "rod_float" for row in fleet),
            },
            "data_source": self.data_dir.name,
            "data_as_of": data_as_of,
            "model_artifact": self.model_path.name,
            "model_trained_at": bundle["trained_at"],
            "model_training_data_dir": bundle["training_data_dir"],
            "model_training_data_fingerprint": bundle["training_data_fingerprint"],
            "data_matches_model_training": (
                dataset_fingerprint(self.data_dir) == bundle["training_data_fingerprint"]
            ),
            "model_versions": bundle["versions"],
            "refreshed_at": refreshed_at,
            "next_refresh_at": None,
            "refresh_interval_seconds": self.refresh_interval_seconds,
            "last_refresh_error": None,
            "alerts": alerts,
        }

    def current(self):
        with self._state_lock:
            return self._state

    def refresh(self):
        if not self._refresh_lock.acquire(blocking=False):
            state = self.current()
            return {"refreshed": False, "refreshing": True,
                    "refreshed_at": state["refreshed_at"]}
        try:
            previous = self.current()
            dataset = TwinDataset.load(str(self.data_dir))
            bundle = load_model_bundle(self.model_path)
            twin = WellDigitalTwin(dataset, model_bundle=bundle)
            refreshed = self._build_state(dataset, twin, bundle, previous)
            refreshed["next_refresh_at"] = (
                datetime.now().astimezone().timestamp() + self.refresh_interval_seconds
            )
            refreshed["next_refresh_at"] = datetime.fromtimestamp(
                refreshed["next_refresh_at"], datetime.now().astimezone().tzinfo
            ).isoformat(timespec="seconds")
            with self._state_lock:
                self._state = refreshed
            self._next_refresh_monotonic = monotonic() + self.refresh_interval_seconds
            print(f"Snapshot refreshed at {refreshed['refreshed_at']}; "
                  f"{len(refreshed['alerts'])} change alerts retained.")
            return {"refreshed": True, **self.metadata(refreshed)}
        except Exception as exc:
            print(f"Snapshot refresh failed: {exc}")
            with self._state_lock:
                stale = dict(self._state)
                stale["last_refresh_error"] = str(exc)
                stale["next_refresh_at"] = self._next_refresh_time()
                self._state = stale
            self._next_refresh_monotonic = monotonic() + self.refresh_interval_seconds
            return {"refreshed": False, "error": "Dataset or saved model artifact reload failed; previous snapshot retained."}
        finally:
            self._refresh_lock.release()

    def metadata(self, state=None):
        state = state or self.current()
        return {key: state[key] for key in (
            "data_source", "data_as_of", "refreshed_at", "next_refresh_at",
            "refresh_interval_seconds", "last_refresh_error", "model_artifact",
            "model_trained_at", "model_training_data_dir",
            "data_matches_model_training", "model_versions",
        )}

    def _next_refresh_time(self):
        return datetime.fromtimestamp(
            datetime.now().astimezone().timestamp() + self.refresh_interval_seconds,
            datetime.now().astimezone().tzinfo,
        ).isoformat(timespec="seconds")

    def start(self):
        if self._worker and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = Thread(target=self._refresh_loop, name="twin-refresh", daemon=True)
        self._worker.start()

    def _refresh_loop(self):
        while not self._stop_event.is_set():
            delay = max(0.0, self._next_refresh_monotonic - monotonic())
            if self._stop_event.wait(delay):
                return
            result = self.refresh()
            if result.get("refreshing"):
                self._stop_event.wait(5)

    def stop(self):
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2)


SERVICE = TwinService(DATA_DIR, MODEL_PATH, REFRESH_INTERVAL_SECONDS)
initial_state = SERVICE.current()
initial_state["next_refresh_at"] = SERVICE._next_refresh_time()


def get_overview() -> dict:
    state = SERVICE.current()
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **SERVICE.metadata(state),
        "summary": state["summary"],
        "fleet": state["fleet"],
        "alerts": state["alerts"],
    }


def get_well_detail(well_id: str) -> dict | None:
    state = SERVICE.current()
    dataset = state["dataset"]
    twin = state["twin"]
    if well_id not in set(dataset.well_master["well_id"].astype(str)):
        return None

    cycles = dataset.css_cycles[dataset.css_cycles.well_id == well_id].sort_values("cycle_number")
    latest_cycle = int(cycles.cycle_number.iloc[-1])
    daily = dataset.daily_production[
        (dataset.daily_production.well_id == well_id)
        & (dataset.daily_production.cycle_number == latest_cycle)
    ].sort_values("day_index")
    srp = dataset.srp_operations[
        (dataset.srp_operations.well_id == well_id)
        & (dataset.srp_operations.cycle_number == latest_cycle)
    ].sort_values("day_index")
    snapshot = twin.srp_snapshot(well_id, latest_cycle, int(daily.day_index.iloc[-1]))

    cards = [card for card in dataset.cards
             if card["well_id"] == well_id and card["cycle_number"] == latest_cycle]
    card_views = []
    for card in cards:
        prediction = twin.card_classifier.predict(card)
        card_views.append({
            "card_id": card["card_id"],
            "day_index": card["day_index"],
            "fault_label": card["fault_label"],
            "prediction": prediction,
        })

    well_master = dataset.well_master.set_index("well_id").loc[well_id].to_dict()
    return json_safe({
        "well_id": well_id,
        "well_master": well_master,
        "latest_cycle": latest_cycle,
        "cycle_history": cycles.to_dict(orient="records"),
        "latest_cycle_daily": daily[[
            "day_index", "wellhead_temp_c", "viscosity_cp", "oil_rate_bopd",
            "pump_fillage_pct", "rod_float_flag",
        ]].to_dict(orient="records"),
        "latest_cycle_srp": srp[[
            "day_index", "spm", "motor_current_a", "impact_load_proxy",
        ]].to_dict(orient="records"),
        "sample_cards": card_views,
        "card_classification": snapshot["card_classification"],
        "current_rod_risk": snapshot["rod_risk"],
        "viscosity_trend": snapshot["viscosity_trend"],
        "current_advice": snapshot["advice"],
    })


class TwinRequestHandler(BaseHTTPRequestHandler):
    server_version = "BaghewalaTwin/1.0"

    def log_message(self, format_string, *args):
        print(f"[{self.log_date_time_string()}] {format_string % args}")

    def send_json(self, payload, status=200):
        body = json.dumps(json_safe(payload), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def serve_static(self, request_path):
        if request_path == "/":
            request_path = "/index.html"
        relative = Path(unquote(request_path).lstrip("/"))
        target = (DASHBOARD_DIR / relative).resolve()
        if DASHBOARD_DIR.resolve() not in target.parents or not target.is_file():
            self.send_json({"error": "Not found"}, status=404)
            return
        content = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/health":
            state = SERVICE.current()
            self.send_json({
                "status": "ok",
                **SERVICE.metadata(state),
                "well_count": int(state["dataset"].well_master.well_id.nunique()),
                "models": [
                    "HistGradientBoostingRegressor",
                    "RandomForestClassifier",
                    "LogisticRegression",
                    "GaussianProcessRegressor",
                ],
            })
        elif path == "/api/overview":
            self.send_json(get_overview())
        elif path == "/api/alerts":
            state = SERVICE.current()
            self.send_json({"alerts": state["alerts"], **SERVICE.metadata(state)})
        elif path == "/api/wells":
            state = SERVICE.current()
            self.send_json({"well_ids": state["dataset"].well_master.well_id.astype(str).tolist()})
        else:
            match = re.fullmatch(r"/api/wells/([^/]+)", path)
            if match:
                detail = get_well_detail(unquote(match.group(1)))
                self.send_json(detail if detail is not None else {"error": "Unknown well"},
                               status=200 if detail is not None else 404)
            else:
                self.serve_static(path)

    def do_POST(self):
        path = urlsplit(self.path).path
        if path == "/api/refresh":
            result = SERVICE.refresh()
            status = 200 if result.get("refreshed") else (202 if result.get("refreshing") else 503)
            self.send_json(result, status=status)
            return

        forecast_match = re.fullmatch(r"/api/wells/([^/]+)/css-forecast", path)
        if forecast_match:
            well_id = unquote(forecast_match.group(1))
            state = SERVICE.current()
            twin = state["twin"]
            if well_id not in set(state["dataset"].well_master["well_id"].astype(str)):
                self.send_json({"error": "Unknown well"}, status=404)
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 4096:
                    raise ValueError("Request body must be between 1 and 4096 bytes.")
                inputs = json.loads(self.rfile.read(content_length))
                if not isinstance(inputs, dict):
                    raise ValueError("Request body must be a JSON object.")
                cycle_params = {}
                for name, (lower, upper) in BOUNDS.items():
                    value = float(inputs[name])
                    if not np.isfinite(value) or not lower <= value <= upper:
                        raise ValueError(f"{name} must be between {lower:g} and {upper:g}.")
                    cycle_params[name] = value
                forecast = twin.forecaster.predict(twin._well_features(well_id), cycle_params)
                self.send_json({
                    "well_id": well_id,
                    "inputs": cycle_params,
                    "forecast": forecast,
                    "model": {
                        "artifact": state["model_artifact"],
                        "trained_at": state["model_trained_at"],
                        "training_data_matches_current": state["data_matches_model_training"],
                    },
                })
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                self.send_json({"error": str(exc)}, status=400)
            except Exception as exc:
                print(f"CSS forecast failed for {well_id}: {exc}")
                self.send_json({"error": "Could not calculate a forecast"}, status=500)
            return

        match = re.fullmatch(r"/api/wells/([^/]+)/css-recommendation", path)
        if not match:
            self.send_json({"error": "Not found"}, status=404)
            return

        well_id = unquote(match.group(1))
        state = SERVICE.current()
        if well_id not in set(state["dataset"].well_master["well_id"].astype(str)):
            self.send_json({"error": "Unknown well"}, status=404)
            return
        try:
            self.send_json(state["twin"].css_recommendation(well_id))
        except Exception as exc:
            print(f"CSS recommendation failed for {well_id}: {exc}")
            self.send_json({"error": "Could not calculate a recommendation"}, status=500)


def main():
    host = os.getenv("BAGHEWALA_HOST", "127.0.0.1")
    port = int(os.getenv("BAGHEWALA_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), TwinRequestHandler)
    SERVICE.start()
    print(f"Baghewala Digital Twin ready at http://{host}:{port}")
    print(f"Data source: {DATA_DIR}")
    print(f"Source files and saved model artifact reload every {REFRESH_INTERVAL_SECONDS // 60} minutes.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping Baghewala Digital Twin server.")
    finally:
        SERVICE.stop()
        server.server_close()


if __name__ == "__main__":
    main()