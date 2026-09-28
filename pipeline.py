"""
pipeline.py
============
The Baghewala Well-to-Surface Digital Twin pipeline, wired end-to-end.

HOW TO READ THIS FILE
----------------------
The pipeline includes data loading, feature engineering, Bayesian
optimization, economic scoring, and dashboard export. GAP 1 uses a pooled
gradient-boosting model; the richer GAP 2 and GAP 3 implementations live in
their own modules and are loaded by the orchestrator. GAP 4 remains a
transparent rule-based advisor.

Each GAP class documents the input/output contract expected by the
orchestrator. The current models are research prototypes trained or
calibrated on synthetic data and require real-data validation before use.

    GAP 1  ReservoirCycleForecaster   -- feeds the CSS Bayesian optimizer
    GAP 2  DynamometerCardClassifier  -- feeds the SRP advisor
    GAP 3  RodFailureRiskModel        -- feeds the SRP advisor
    GAP 4  SRPAdvisor                 -- consumes GAP 2 + GAP 3 output

Pipeline flow (matches the architecture diagram):
    data layer -> GAP 1 -> Bayesian optimizer -> CSS recommendation
    data layer -> GAP 2 -> GAP 4 \\
    data layer -> GAP 3 -> GAP 4 /-> SRP recommendation
    (all of the above) -> fleet snapshot -> dashboard export
"""

import json
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)  # sklearn GP hyperparameter-bound chatter
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baghewala_dataset")

# Illustrative economics -- replace with real OIL India figures.
OIL_PRICE_INR_PER_BBL = 5800.0
STEAM_COST_INR_PER_BBL_CWE = 950.0
WORKOVER_COST_INR = 450_000.0

# Assumed viscosity fraction (of a well's reservoir-temperature viscosity)
# above which rod-float risk starts rising. Used only to connect the CSS
# optimizer to expected mechanical wear -- see _predict_rod_float_days().
# This is a global placeholder assumption, not a fitted value; calibrate it
# per well against real fillage/rod-float history once you have it.
ROD_FLOAT_VISCOSITY_FRACTION = 0.65


# =============================================================================
# DATA LAYER  (fully implemented)
# =============================================================================
@dataclass
class TwinDataset:
    well_master: pd.DataFrame
    pvt: pd.DataFrame
    css_cycles: pd.DataFrame
    daily_production: pd.DataFrame
    srp_operations: pd.DataFrame
    cards: list
    rod_failures: pd.DataFrame

    @classmethod
    def load(cls, data_dir=DATA_DIR):
        with open(os.path.join(data_dir, "dynamometer_cards.json")) as f:
            cards = json.load(f)
        return cls(
            well_master=pd.read_csv(os.path.join(data_dir, "well_master.csv")),
            pvt=pd.read_csv(os.path.join(data_dir, "pvt_viscosity_samples.csv")),
            css_cycles=pd.read_csv(os.path.join(data_dir, "css_cycles.csv")),
            daily_production=pd.read_csv(os.path.join(data_dir, "daily_production.csv")),
            srp_operations=pd.read_csv(os.path.join(data_dir, "srp_operations.csv")),
            cards=cards,
            rod_failures=pd.read_csv(os.path.join(data_dir, "rod_failures.csv")),
        )


def fit_viscosity_model(pvt_samples_for_well: pd.DataFrame):
    """Fully implemented (not a gap): fits ln(eta) = A + B/T from a well's
    PVT lab samples via ordinary least squares. This is the calibration
    step the architecture doc calls the 'wellbore/viscosity model'."""
    t_kelvin = pvt_samples_for_well["temperature_c"].values + 273.15
    y = np.log(pvt_samples_for_well["viscosity_cp"].values)
    x = (1 / t_kelvin).reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    B = reg.coef_[0]
    A = reg.intercept_
    return A, B


def andrade_viscosity(temp_c, A, B):
    t_kelvin = np.asarray(temp_c, dtype=float) + 273.15
    return np.exp(A + B / t_kelvin)


# =============================================================================
# GAP 1 -- Reservoir / Cycle-Performance Forecaster
# =============================================================================
class ReservoirCycleForecaster:
    """
    GAP 1 of 4.

    Predicts what a CANDIDATE CSS cycle design would do to a given well,
    before you actually inject the steam. This is the surrogate objective
    the Bayesian optimizer below searches over -- it gets called dozens of
    times per well, so it needs to be cheap to evaluate.

    ---------------------------------------------------------------------
    INPUT to `.predict(well_features, cycle_params)`:

        well_features: dict
            {
              "api_gravity": float,                       # deg API
              "reservoir_temp_c": float,                   # deg C
              "reservoir_depth_m": float,
              "viscosity_at_reservoir_temp_cp": float,      # cP
              "cycle_number": int,                          # which cycle this well is on
              "cumulative_steam_to_date_bbl": float,
              "prior_cycle_sor": float,                     # SOR of the well's most recent cycle (NaN if first cycle)
            }

        cycle_params: dict  (the CANDIDATE the optimizer wants scored)
            {
              "steam_volume_cwe_bbl": float,   # 2000-8000 typical
              "injection_pressure_kpa": float, # 5000-10000 typical
              "soak_time_days": float,         # 2-10 typical
            }

    OUTPUT: dict, ALL keys required
        {
          "peak_post_soak_temp_c": float,
          "thermal_decay_rate_per_day": float,   # for T(t) = T_res + (T_peak-T_res)*exp(-rate*t)
          "predicted_cycle_oil_bbl": float,
          "predicted_cycle_sor": float,          # bbl steam (CWE) / bbl oil
          "predicted_production_days": float,
        }
    ---------------------------------------------------------------------
    IMPLEMENTATION: one HistGradientBoostingRegressor per target, trained
    across wells with a one-hot well identity feature. Unknown well IDs map
    to an all-zero identity vector, allowing predictions for unseen wells.
    Small leaves and a low learning rate are used because the sample has only
    126 cycles across 18 wells.
    """

    FEATURE_COLS = [
        "api_gravity", "reservoir_temp_c", "reservoir_depth_m",
        "viscosity_at_reservoir_temp_cp", "cycle_number",
        "cumulative_steam_to_date_bbl",
        "steam_volume_cwe_bbl", "injection_pressure_kpa", "soak_time_days",
    ]
    TARGET_COLS = [
        "peak_post_soak_temp_c", "thermal_decay_rate_per_day",
        "cycle_cum_oil_bbl", "cycle_sor", "production_days",
    ]

    def __init__(self, dataset: TwinDataset):
        merged = dataset.css_cycles.merge(dataset.well_master, on="well_id")
        self._well_ids = sorted(merged["well_id"].astype(str).unique())
        X = self._make_features(merged[self.FEATURE_COLS], merged["well_id"])
        self._models = {}
        for target in self.TARGET_COLS:
            model = HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_iter=200,
                max_leaf_nodes=7,
                min_samples_leaf=3,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=0,
            )
            self._models[target] = model.fit( X, merged[target].to_numpy(dtype=float))

    def _make_features(self, numeric_features, well_ids):
        numeric = numeric_features.fillna(0).to_numpy(dtype=float)
        well_ids = np.asarray(well_ids, dtype=str)
        well_identity = np.column_stack([well_ids == well_id for well_id in self._well_ids]).astype(float)
        return np.column_stack([numeric, well_identity])

    def predict(self, well_features: dict, cycle_params: dict) -> dict:
        row = pd.DataFrame([[
            well_features["api_gravity"], well_features["reservoir_temp_c"],
            well_features["reservoir_depth_m"], well_features["viscosity_at_reservoir_temp_cp"],
            well_features["cycle_number"], well_features["cumulative_steam_to_date_bbl"],
            cycle_params["steam_volume_cwe_bbl"], cycle_params["injection_pressure_kpa"],
            cycle_params["soak_time_days"],
        ]], columns=self.FEATURE_COLS)
        well_id = well_features.get("well_id", "")
        features = self._make_features(row, [well_id])
        pred = {target: float(model.predict(features)[0])
                for target, model in self._models.items()}
        return {
            "peak_post_soak_temp_c": pred["peak_post_soak_temp_c"],
            "thermal_decay_rate_per_day": max(pred["thermal_decay_rate_per_day"], 1e-4),
            "predicted_cycle_oil_bbl": max(pred["cycle_cum_oil_bbl"], 0.0),
            "predicted_cycle_sor": max(pred["cycle_sor"], 0.01),
            "predicted_production_days": max(pred["production_days"], 1.0),
        }


# =============================================================================
# GAP 2 -- Dynamometer Card Fault Classifier
# =============================================================================
FAULT_LABELS = ["normal", "rod_float", "fluid_pound", "gas_interference", "worn_valve", "parted_rod"]


def _extract_card_features(position, load):
    """Fully implemented (not the gap itself): turns a raw (position, load)
    card into a small set of hand-crafted numeric features. A real deep
    model (Gap 2 replacement) would more likely consume the raw resampled
    arrays directly (e.g. as a 1D-CNN input) instead of these features."""
    position = np.asarray(position)
    load = np.asarray(load)
    half = len(position) // 2
    up_load, down_load = load[:half], load[half:]
    l_min, l_max = load.min(), load.max()
    card_range = max(l_max - l_min, 1e-6)
    trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renamed trapz -> trapezoid
    area = trapz(up_load, position[:half]) - trapz(down_load, position[half:][::-1])
    down_grad = np.gradient(down_load)
    steepest_drop_frac = np.argmin(down_grad) / max(len(down_grad) - 1, 1)
    return [
        card_range,
        area / card_range,
        (down_load.mean() - up_load.mean()) / card_range,
        steepest_drop_frac,
        down_load.std() / card_range,
    ]


class DynamometerCardClassifier:
    """
    GAP 2 of 4.

    Classifies a single dynamometer card's operating condition.

    ---------------------------------------------------------------------
    INPUT to `.predict(card)`:

        card: dict
            {
              "position": list[float],   # length N, 0->1->0 over one stroke
              "load": list[float],       # length N, same length as position [lbf]
              "spm": float,               # optional context
              "motor_current_a": float,   # optional context
            }

    OUTPUT: dict, ALL keys required
        {
          "predicted_label": str,             # one of FAULT_LABELS
          "confidence": float,                 # 0-1, confidence in predicted_label
          "class_probabilities": dict[str, float],  # one entry per FAULT_LABELS, sums to ~1
        }
    ---------------------------------------------------------------------
    PLACEHOLDER IMPLEMENTATION: a RandomForestClassifier trained on five
    hand-crafted summary features (card area, load asymmetry, where the
    downstroke drop happens, etc.) -- i.e. the "classical ML with manual
    feature extraction" tier described in the SRP-diagnostics literature.

    REPLACE WITH: a CNN or 1D-CNN trained directly on the raw
    (position, load) sequences -- the "deep learning with automatic
    feature extraction" tier, which is what the cited papers show
    consistently outperforms hand-crafted features once you have enough
    labeled cards (real or wave-equation-simulated).
    """

    def __init__(self, dataset: TwinDataset):
        X = [_extract_card_features(c["position"], c["load"]) for c in dataset.cards]
        y = [c["fault_label"] for c in dataset.cards]
        self._model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0)
        self._model.fit(X, y)
        self._classes = list(self._model.classes_)

    def predict(self, card: dict) -> dict:
        feats = [_extract_card_features(card["position"], card["load"])]
        probs = self._model.predict_proba(feats)[0]
        class_probabilities = {c: float(p) for c, p in zip(self._classes, probs)}
        for label in FAULT_LABELS:
            class_probabilities.setdefault(label, 0.0)
        best_label = max(class_probabilities, key=class_probabilities.get)
        return {
            "predicted_label": best_label,
            "confidence": class_probabilities[best_label],
            "class_probabilities": class_probabilities,
        }


# =============================================================================
# GAP 3 -- Rod Failure Risk Model
# =============================================================================
class RodFailureRiskModel:
    """
    GAP 3 of 4.

    Scores how likely a well's rod string is to fail soon.

    ---------------------------------------------------------------------
    INPUT to `.predict(rod_state)`:

        rod_state: dict
            {
              "rod_age_days": float,               # since last replacement/completion
              "cumulative_rod_float_days": float,   # rod-float-flagged days since last replacement
              "avg_impact_load_proxy_last_30d": float,
              "rod_material": str,                  # "steel_grade_D" / "steel_grade_EL" / "fiberglass_hybrid"
              "depth_m": float,
            }

    OUTPUT: dict, ALL keys required
        {
          "failure_probability_30d": float,   # 0-1
          "risk_tier": str,                    # "low" / "medium" / "high"
        }
    ---------------------------------------------------------------------
    PLACEHOLDER IMPLEMENTATION: a hand-tuned logistic formula over rod age
    and cumulative rod-float exposure -- no training, deliberately naive.

    REPLACE WITH: a model actually fit on rod_failures.csv (survival
    analysis / Cox proportional hazards, or a Weibull time-to-failure
    regression, per the fatigue-life literature cited earlier) using
    real positive AND negative (non-failure) examples constructed from
    srp_operations.csv.
    """

    MATERIAL_RISK_MULT = {"steel_grade_D": 1.0, "steel_grade_EL": 0.85, "fiberglass_hybrid": 0.6}

    def predict(self, rod_state: dict) -> dict:
        age_term = 0.0006 * rod_state["rod_age_days"]
        float_term = 0.02 * rod_state["cumulative_rod_float_days"]
        impact_term = 0.15 * rod_state["avg_impact_load_proxy_last_30d"]
        material_mult = self.MATERIAL_RISK_MULT.get(rod_state["rod_material"], 1.0)
        logit = -9.5 + material_mult * (age_term + float_term + impact_term)
        prob = 1 / (1 + np.exp(-logit))
        prob_30d = float(1 - (1 - prob) ** 30)  # convert daily hazard-ish rate to a 30-day probability
        tier = "high" if prob_30d > 0.15 else ("medium" if prob_30d > 0.04 else "low")
        return {"failure_probability_30d": min(prob_30d, 0.99), "risk_tier": tier}


# =============================================================================
# GAP 4 -- SRP Speed & Stroke Advisor
# =============================================================================
class SRPAdvisor:
    """
    GAP 4 of 4.

    Turns the card classification (Gap 2) and rod risk score (Gap 3) into
    a concrete SPM/stroke recommendation with a plain-English reason.

    ---------------------------------------------------------------------
    INPUT to `.advise(state)`:

        state: dict
            {
              "current_spm": float,
              "current_stroke_length_in": float,
              "card_classification": dict,     # exact output of DynamometerCardClassifier.predict()
              "rod_risk": dict,                # exact output of RodFailureRiskModel.predict()
              "viscosity_trend": str,          # "rising" / "falling" / "stable"
            }

    OUTPUT: dict, ALL keys required
        {
          "recommended_spm": float,
          "recommended_stroke_length_in": float,
          "action": str,           # "reduce_speed" / "hold" / "increase_speed"
          "rationale": str,        # one sentence, human-readable
        }
    ---------------------------------------------------------------------
    PLACEHOLDER IMPLEMENTATION: a transparent rule table. This is
    deliberately NOT a black box for a first deployment -- an operator
    should be able to read the rationale and agree or override.

    REPLACE / UPGRADE WITH: the intra-stroke VFD-frequency deep-RL
    approach or the rod-well-reservoir model-predictive-control approach
    from the architecture doc, once the rule engine has a track record
    and you're ready to trust a less-interpretable controller.
    """

    def advise(self, state: dict) -> dict:
        label = state["card_classification"]["predicted_label"]
        confidence = state["card_classification"]["confidence"]
        risk_tier = state["rod_risk"]["risk_tier"]
        spm, stroke = state["current_spm"], state["current_stroke_length_in"]

        if label == "rod_float" and confidence > 0.5:
            new_spm = round(spm * 0.8, 2)
            return {
                "recommended_spm": new_spm,
                "recommended_stroke_length_in": stroke,
                "action": "reduce_speed",
                "rationale": (
                    f"Card classifier flagged rod float (confidence {confidence:.0%}); "
                    f"cutting SPM from {spm:.2f} to {new_spm:.2f} to let the rod string "
                    f"catch up before it impacts on the downstroke."
                ),
            }
        if risk_tier == "high":
            new_spm = round(spm * 0.9, 2)
            return {
                "recommended_spm": new_spm,
                "recommended_stroke_length_in": stroke,
                "action": "reduce_speed",
                "rationale": (
                    f"Rod-failure risk model reports HIGH risk "
                    f"({state['rod_risk']['failure_probability_30d']:.0%} in 30 days); "
                    f"reducing SPM as a precaution while a workover is scheduled."
                ),
            }
        if state["viscosity_trend"] == "rising" and label == "normal":
            return {
                "recommended_spm": spm,
                "recommended_stroke_length_in": stroke,
                "action": "hold",
                "rationale": (
                    "Viscosity is trending up as the well cools post-cycle but the card "
                    "still reads normal -- holding current settings and watching closely."
                ),
            }
        if label == "normal" and risk_tier == "low" and state["viscosity_trend"] == "falling":
            new_spm = round(spm * 1.05, 2)
            return {
                "recommended_spm": new_spm,
                "recommended_stroke_length_in": stroke,
                "action": "increase_speed",
                "rationale": (
                    "Fresh post-steam viscosity is low and the card reads normal -- "
                    "a small speed increase can capture more of the high-rate window."
                ),
            }
        return {
            "recommended_spm": spm,
            "recommended_stroke_length_in": stroke,
            "action": "hold",
            "rationale": "No clear signal to change settings; holding current operation.",
        }


# =============================================================================
# BAYESIAN CSS CYCLE OPTIMIZER  (fully implemented -- wraps GAP 1)
# =============================================================================
BOUNDS = {
    "steam_volume_cwe_bbl": (2500.0, 8000.0),
    "injection_pressure_kpa": (5500.0, 10000.0),
    "soak_time_days": (2.0, 9.0),
}


def _predict_rod_float_days(forecast: dict, well_features: dict, andrade_params: tuple) -> float:
    """Fully implemented (not a gap): walks the SAME exponential decay curve
    Gap 1 predicts -- T(t) = T_res + (T_peak-T_res)*exp(-rate*t) -- through
    the well's own fitted Andrade viscosity model, and counts how many of
    the predicted production days would fall above the assumed rod-float
    viscosity threshold. This is what lets the CSS optimizer see the
    mechanical-wear side effect of a candidate steam schedule, not just its
    oil/steam economics -- without this, GAP 1's thermal_decay_rate output
    and WORKOVER_COST_INR below would otherwise never be used."""
    A, B = andrade_params
    res_temp = well_features["reservoir_temp_c"]
    res_visc = well_features["viscosity_at_reservoir_temp_cp"]
    threshold = res_visc * ROD_FLOAT_VISCOSITY_FRACTION
    n_days = max(int(round(forecast["predicted_production_days"])), 1)
    days = np.arange(n_days)
    temps = res_temp + (forecast["peak_post_soak_temp_c"] - res_temp) * np.exp(
        -forecast["thermal_decay_rate_per_day"] * days
    )
    viscs = andrade_viscosity(temps, A, B)
    return float(np.sum(viscs > threshold))


def _economic_value(forecast: dict, cycle_params: dict, *, rod_risk_model,
                     well_features: dict, andrade_params: tuple, current_rod_state: dict) -> dict:
    """Returns the full cost/benefit breakdown (not just a scalar) so the
    optimizer's search history keeps a record of WHY a candidate scored the
    way it did -- revenue and steam cost as before, PLUS an expected
    workover cost priced in from the candidate's projected rod-float
    exposure. `value` is the single number the optimizer maximizes."""
    revenue = forecast["predicted_cycle_oil_bbl"] * OIL_PRICE_INR_PER_BBL
    steam_cost = cycle_params["steam_volume_cwe_bbl"] * STEAM_COST_INR_PER_BBL_CWE

    predicted_float_days = _predict_rod_float_days(forecast, well_features, andrade_params)
    projected_rod_state = dict(current_rod_state)
    projected_rod_state["cumulative_rod_float_days"] = (
        current_rod_state["cumulative_rod_float_days"] + predicted_float_days
    )
    projected_rod_state["rod_age_days"] = (
        current_rod_state["rod_age_days"] + forecast["predicted_production_days"]
    )
    # NOTE: avg_impact_load_proxy_last_30d is carried forward unchanged from
    # the well's current state -- Gap 1 doesn't forecast this signal, so this
    # is a deliberate simplification, not a prediction.
    risk = rod_risk_model.predict(projected_rod_state)
    expected_workover_cost = risk["failure_probability_30d"] * WORKOVER_COST_INR

    return {
        "value": revenue - steam_cost - expected_workover_cost,
        "revenue_inr": revenue,
        "steam_cost_inr": steam_cost,
        "predicted_rod_float_days": predicted_float_days,
        "expected_workover_cost_inr": expected_workover_cost,
        "projected_failure_probability_30d": risk["failure_probability_30d"],
    }


def recommend_next_css_cycle(forecaster: ReservoirCycleForecaster, well_features: dict,
                              rod_risk_model, andrade_params: tuple, current_rod_state: dict,
                              n_initial=10, n_iter=20, seed=0) -> dict:
    """Fully implemented Bayesian optimization loop (Gaussian Process +
    expected improvement) searching (steam_volume, injection_pressure,
    soak_time) for the candidate that maximizes predicted economic value --
    revenue minus steam cost minus expected workover cost -- using
    `forecaster` (GAP 1) as the cheap surrogate objective. This is the
    piece the architecture doc calls the 'CSS parameter recommender', and
    it's also where the CSS-optimizer/rod-mechanics coupling actually
    happens: `rod_risk_model` (GAP 3) is called from inside the objective."""
    rng = np.random.default_rng(seed)
    names = list(BOUNDS.keys())
    lo = np.array([BOUNDS[n][0] for n in names])
    hi = np.array([BOUNDS[n][1] for n in names])

    def sample(n):
        return lo + rng.random((n, len(names))) * (hi - lo)

    def evaluate(x_row):
        params = dict(zip(names, x_row))
        forecast = forecaster.predict(well_features, params)
        econ = _economic_value(forecast, params, rod_risk_model=rod_risk_model,
                                well_features=well_features, andrade_params=andrade_params,
                                current_rod_state=current_rod_state)
        return econ["value"], forecast, params, econ

    X, y, history = [], [], []
    for x_row in sample(n_initial):
        val, forecast, params, econ = evaluate(x_row)
        X.append(x_row)
        y.append(val)
        history.append({"params": params, "forecast": forecast, "value": val, "economics": econ})

    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e5)
    for _ in range(n_iter):
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2)
        gp.fit(np.array(X), np.array(y))

        candidates = sample(500)
        mu, sigma = gp.predict(candidates, return_std=True)
        best_y = max(y)
        with np.errstate(divide="ignore"):
            z = (mu - best_y) / np.where(sigma > 1e-9, sigma, 1e-9)
            ei = (mu - best_y) * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-9] = 0.0
        next_x = candidates[np.argmax(ei)]

        val, forecast, params, econ = evaluate(next_x)
        X.append(next_x)
        y.append(val)
        history.append({"params": params, "forecast": forecast, "value": val, "economics": econ})

    best_idx = int(np.argmax(y))
    return {"recommended_params": history[best_idx]["params"],
            "predicted_forecast": history[best_idx]["forecast"],
            "predicted_economic_value_inr": history[best_idx]["value"],
            "economics_breakdown": history[best_idx]["economics"],
            "search_history": history}


# =============================================================================
# ORCHESTRATOR  (fully implemented)
# =============================================================================
class WellDigitalTwin:
    def __init__(self, dataset: TwinDataset, model_bundle: dict | None = None):
        self.data = dataset
        if model_bundle is None:
            from gap2_card_classifier import DynamometerCardClassifier
            from gap3_rod_risk_model import build_panel, RodFailureRiskModel

            self.forecaster = ReservoirCycleForecaster(dataset)
            self.card_classifier = DynamometerCardClassifier(dataset.cards, dataset.well_master)
            self.rod_risk_model = RodFailureRiskModel(build_panel(dataset))
        else:
            models = model_bundle["models"]
            self.forecaster = models["gap1_forecaster"]
            self.card_classifier = models["gap2_card_classifier"]
            self.rod_risk_model = models["gap3_rod_risk_model"]
        self.srp_advisor = SRPAdvisor()
        self._andrade_cache = {}

    def _andrade_params_for_well(self, well_id: str) -> tuple:
        """Fits (A, B) once per well from pvt_viscosity_samples.csv via
        fit_viscosity_model(), and caches it -- this is the calibration step
        that was previously dead code; it's now what lets the CSS optimizer
        translate a predicted temperature trajectory into a viscosity, and
        therefore a rod-float-days, trajectory."""
        if well_id not in self._andrade_cache:
            pvt_well = self.data.pvt[self.data.pvt.well_id == well_id]
            self._andrade_cache[well_id] = fit_viscosity_model(pvt_well)
        return self._andrade_cache[well_id]

    def _rod_state_at(self, well_id: str, cycle_number: int, day_index: int) -> dict:
        """Rod age / cumulative rod-float exposure / recent impact-load
        average as of a specific (cycle, day) in a well's history, with
        exposure resetting at the most recent rod-string replacement (the
        most recent failure event before this point), if any. Shared by
        srp_snapshot() (the well's CURRENT state) and css_recommendation()
        (current state, before projecting a candidate cycle's effect on it)
        so the two can't drift out of sync with each other.

        The 30-day SRP window is built by explicitly sorting on
        (cycle_number, day_index) before taking the tail -- day_index resets
        to 0 every new cycle, so a plain `day_index <= day_index` filter
        would otherwise also match rows from OTHER cycles, including ones
        that haven't happened yet relative to this point."""
        w = self.data.well_master.set_index("well_id").loc[well_id]
        daily_all = self.data.daily_production
        history_to_date = daily_all[(daily_all.well_id == well_id)
                                     & ((daily_all.cycle_number < cycle_number)
                                        | ((daily_all.cycle_number == cycle_number) & (daily_all.day_index <= day_index)))]

        past_failures = self.data.rod_failures[
            (self.data.rod_failures.well_id == well_id)
            & ((self.data.rod_failures.cycle_number_at_failure < cycle_number)
               | ((self.data.rod_failures.cycle_number_at_failure == cycle_number)
                  & (self.data.rod_failures.day_index_at_failure <= day_index)))
        ].sort_values(["cycle_number_at_failure", "day_index_at_failure"])
        if len(past_failures):
            last_failure_cycle = int(past_failures.iloc[-1]["cycle_number_at_failure"])
            last_failure_day = int(past_failures.iloc[-1]["day_index_at_failure"])
            history_since_replacement = history_to_date[
                (history_to_date.cycle_number > last_failure_cycle)
                | ((history_to_date.cycle_number == last_failure_cycle) & (history_to_date.day_index > last_failure_day))
            ]
        else:
            history_since_replacement = history_to_date

        srp_all = self.data.srp_operations
        srp_history_to_date = srp_all[(srp_all.well_id == well_id)
                                       & ((srp_all.cycle_number < cycle_number)
                                          | ((srp_all.cycle_number == cycle_number) & (srp_all.day_index <= day_index)))]
        recent = srp_history_to_date.sort_values(["cycle_number", "day_index"]).tail(30)

        return {
            "rod_age_days": int(len(history_since_replacement)),
            "cumulative_rod_float_days": int(history_since_replacement["rod_float_flag"].sum()),
            "avg_impact_load_proxy_last_30d": float(recent["impact_load_proxy"].mean()) if len(recent) else 0.0,
            "rod_material": w["rod_material"],
            "depth_m": float(w["pump_setting_depth_m"]),
        }

    def _well_features(self, well_id):
        w = self.data.well_master.set_index("well_id").loc[well_id]
        cycles = self.data.css_cycles[self.data.css_cycles.well_id == well_id].sort_values("cycle_number")
        last = cycles.iloc[-1]
        return {
            "well_id": well_id,
            "api_gravity": w["api_gravity"],
            "reservoir_temp_c": w["reservoir_temp_c"],
            "reservoir_depth_m": w["reservoir_depth_m"],
            "viscosity_at_reservoir_temp_cp": w["viscosity_at_reservoir_temp_cp"],
            "cycle_number": int(last["cycle_number"]) + 1,
            "cumulative_steam_to_date_bbl": float(last["cumulative_steam_to_date_bbl"]),
            "prior_cycle_sor": float(last["cycle_sor"]),
        }

    def css_recommendation(self, well_id: str) -> dict:
        wf = self._well_features(well_id)
        cycles = self.data.css_cycles[self.data.css_cycles.well_id == well_id]
        last_cycle_number = int(cycles.cycle_number.max())
        last_day = int(self.data.daily_production[
            (self.data.daily_production.well_id == well_id)
            & (self.data.daily_production.cycle_number == last_cycle_number)
        ].day_index.max())
        rod_state_now = self._rod_state_at(well_id, last_cycle_number, last_day)
        andrade_params = self._andrade_params_for_well(well_id)
        return recommend_next_css_cycle(self.forecaster, wf, self.rod_risk_model,
                                         andrade_params, rod_state_now)

    def srp_snapshot(self, well_id: str, cycle_number: int, day_index: int) -> dict:
        srp = self.data.srp_operations
        row = srp[(srp.well_id == well_id) & (srp.cycle_number == cycle_number)
                   & (srp.day_index == day_index)].iloc[0]
        matching_cards = [c for c in self.data.cards
                           if c["well_id"] == well_id and c["cycle_number"] == cycle_number]
        card = min(matching_cards, key=lambda c: abs(c["day_index"] - day_index)) if matching_cards else None
        card_result = self.card_classifier.predict(card) if card else None

        rod_state = self._rod_state_at(well_id, cycle_number, day_index)
        risk_result = self.rod_risk_model.predict(rod_state)

        daily = self.data.daily_production
        d_recent = daily[(daily.well_id == well_id) & (daily.cycle_number == cycle_number)
                          & (daily.day_index <= day_index)].sort_values("day_index").tail(5)
        trend = "stable"
        if len(d_recent) >= 2:
            delta = d_recent["viscosity_cp"].iloc[-1] - d_recent["viscosity_cp"].iloc[0]
            trend = "rising" if delta > 50 else ("falling" if delta < -50 else "stable")

        advice = self.srp_advisor.advise({
            "current_spm": float(row["spm"]),
            "current_stroke_length_in": float(row["stroke_length_in"]),
            "card_classification": card_result or {"predicted_label": "unknown", "confidence": 0.0,
                                                     "class_probabilities": {}},
            "rod_risk": risk_result,
            "viscosity_trend": trend,
        })
        return {"card": card, "card_classification": card_result, "rod_risk": risk_result,
                "viscosity_trend": trend, "advice": advice}

    def fleet_snapshot(self) -> list:
        """One summary row per well for the dashboard's fleet table."""
        out = []
        for well_id in self.data.well_master.well_id:
            cycles = self.data.css_cycles[self.data.css_cycles.well_id == well_id].sort_values("cycle_number")
            last_cycle = cycles.iloc[-1]
            srp_last = self.data.srp_operations[
                (self.data.srp_operations.well_id == well_id)
                & (self.data.srp_operations.cycle_number == last_cycle.cycle_number)
            ].sort_values("day_index").iloc[-1]
            daily_last = self.data.daily_production[
                (self.data.daily_production.well_id == well_id)
                & (self.data.daily_production.cycle_number == last_cycle.cycle_number)
            ].sort_values("day_index").iloc[-1]
            snap = self.srp_snapshot(well_id, int(last_cycle.cycle_number), int(daily_last.day_index))
            out.append({
                "well_id": well_id,
                "latest_cycle": int(last_cycle.cycle_number),
                "latest_sor": float(last_cycle.cycle_sor),
                "latest_fillage_pct": float(daily_last.pump_fillage_pct),
                "latest_viscosity_cp": float(daily_last.viscosity_cp),
                "card_label": snap["card_classification"]["predicted_label"] if snap["card_classification"] else "n/a",
                "risk_tier": snap["rod_risk"]["risk_tier"],
                "failure_probability_30d": snap["rod_risk"]["failure_probability_30d"],
                "srp_action": snap["advice"]["action"],
            })
        return out


    def well_detail(self, well_id: str) -> dict:
        """Everything the dashboard needs to render one well's detail panel:
        the latest cycle's daily time series, a few sample dynamometer cards
        with classification overlaid, the CSS recommendation for the next
        cycle, and the current SRP advisory."""
        cycles = self.data.css_cycles[self.data.css_cycles.well_id == well_id].sort_values("cycle_number")
        last_cycle = cycles.iloc[-1]
        daily = self.data.daily_production[
            (self.data.daily_production.well_id == well_id)
            & (self.data.daily_production.cycle_number == last_cycle.cycle_number)
        ].sort_values("day_index")
        srp = self.data.srp_operations[
            (self.data.srp_operations.well_id == well_id)
            & (self.data.srp_operations.cycle_number == last_cycle.cycle_number)
        ].sort_values("day_index")

        sample_cards = [c for c in self.data.cards
                         if c["well_id"] == well_id and c["cycle_number"] == last_cycle.cycle_number]
        card_views = []
        for c in sample_cards:
            result = self.card_classifier.predict(c)
            card_views.append({**c, "prediction": result})

        latest_day = int(daily.day_index.iloc[-1])
        snapshot = self.srp_snapshot(well_id, int(last_cycle.cycle_number), latest_day)

        return {
            "well_id": well_id,
            "well_master": self.data.well_master.set_index("well_id").loc[well_id].to_dict(),
            "cycle_history": cycles.to_dict(orient="records"),
            "latest_cycle_daily": daily[["day_index", "wellhead_temp_c", "viscosity_cp",
                                         "oil_rate_bopd", "pump_fillage_pct", "rod_float_flag"]]
                                   .to_dict(orient="records"),
            "latest_cycle_srp": srp[["day_index", "spm", "motor_current_a", "impact_load_proxy"]]
                                 .to_dict(orient="records"),
            "sample_cards": card_views,
            "css_recommendation": self.css_recommendation(well_id),
            "current_advice": snapshot["advice"],
            "current_rod_risk": snapshot["rod_risk"],
        }

    def export_dashboard_json(self, path: str, detail_well_ids=None):
        """Writes the single JSON file dashboard.html is built to consume:
        the full fleet snapshot plus a detailed drill-down for a handful
        of wells (all wells if detail_well_ids is None)."""
        if detail_well_ids is None:
            detail_well_ids = list(self.data.well_master.well_id)

        def clean(obj):
            """Recursively convert numpy scalar types to plain Python so json.dump doesn't choke."""
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [clean(v) for v in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return obj

        payload = {
            "generated_at": pd.Timestamp.now().isoformat(),
            "fleet": clean(self.fleet_snapshot()),
            "wells": {wid: clean(self.well_detail(wid)) for wid in detail_well_ids},
        }
        with open(path, "w") as f:
            json.dump(payload, f)
        return payload


if __name__ == "__main__":
    dataset = TwinDataset.load()
    twin = WellDigitalTwin(dataset)

    print("=== Fleet snapshot ===")
    fleet = twin.fleet_snapshot()
    for row in sorted(fleet, key=lambda r: -r["failure_probability_30d"])[:5]:
        print(row)

    demo_well = fleet[0]["well_id"]
    print(f"\n=== CSS recommendation for {demo_well} ===")
    css_rec = twin.css_recommendation(demo_well)
    print("Recommended params:", css_rec["recommended_params"])
    print("Predicted forecast:", css_rec["predicted_forecast"])
    print("Predicted economic value (INR):", round(css_rec["predicted_economic_value_inr"]))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_data.json")
    print(f"\nExporting dashboard data for all wells to {out_path} ...")
    twin.export_dashboard_json(out_path)
    print("Done.")
