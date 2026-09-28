"""Held-out-well evaluation and CSS optimizer comparison for GAP 1."""

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime
import io
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from pipeline import (
    DATA_DIR,
    ReservoirCycleForecaster,
    TwinDataset,
    recommend_next_css_cycle,
)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class LinearRegressionBaseline:
    """Reproduces the original pooled GAP 1 model for comparison."""

    def __init__(self, dataset):
        merged = dataset.css_cycles.merge(dataset.well_master, on="well_id")
        self.feature_cols = ReservoirCycleForecaster.FEATURE_COLS
        self.target_cols = ReservoirCycleForecaster.TARGET_COLS
        self.model = LinearRegression().fit(
            merged[self.feature_cols].fillna(0).to_numpy(),
            merged[self.target_cols].to_numpy(),
        )

    def predict(self, well_features, cycle_params):
        row = [[
            well_features["api_gravity"], well_features["reservoir_temp_c"],
            well_features["reservoir_depth_m"], well_features["viscosity_at_reservoir_temp_cp"],
            well_features["cycle_number"], well_features["cumulative_steam_to_date_bbl"],
            cycle_params["steam_volume_cwe_bbl"], cycle_params["injection_pressure_kpa"],
            cycle_params["soak_time_days"],
        ]]
        pred = self.model.predict(row)[0]
        return {
            "peak_post_soak_temp_c": float(pred[0]),
            "thermal_decay_rate_per_day": max(float(pred[1]), 1e-4),
            "predicted_cycle_oil_bbl": max(float(pred[2]), 0.0),
            "predicted_cycle_sor": max(float(pred[3]), 0.01),
            "predicted_production_days": max(float(pred[4]), 1.0),
        }


def evaluate_forecasters(dataset, n_splits=5):
    """Compare gradient boosting against the original linear model by well."""
    merged = dataset.css_cycles.merge(dataset.well_master, on="well_id").reset_index(drop=True)
    groups = merged["well_id"].to_numpy()
    unique_wells = np.unique(groups)
    if len(unique_wells) < 2:
        raise ValueError("At least two wells are required for held-out-well evaluation.")

    splitter = GroupKFold(n_splits=min(n_splits, len(unique_wells)))
    targets = ReservoirCycleForecaster.TARGET_COLS
    actual = merged[targets].to_numpy(dtype=float)
    pred_upgraded = np.full_like(actual, np.nan)
    pred_baseline = np.full_like(actual, np.nan)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(merged, groups=groups), start=1):
        train, test = merged.iloc[train_idx], merged.iloc[test_idx]
        train_wells = set(train["well_id"])
        train_dataset = replace(
            dataset,
            well_master=dataset.well_master[dataset.well_master.well_id.isin(train_wells)],
            css_cycles=dataset.css_cycles[dataset.css_cycles.well_id.isin(train_wells)],
        )
        fold_upgraded = ReservoirCycleForecaster(train_dataset)
        fold_baseline = LinearRegressionBaseline(train_dataset)
        X_upgraded = fold_upgraded._make_features(test[fold_upgraded.FEATURE_COLS], test["well_id"])
        X_baseline = test[fold_baseline.feature_cols].fillna(0).to_numpy()
        pred_upgraded[test_idx] = np.column_stack([
            fold_upgraded._models[target].predict(X_upgraded) for target in targets
        ])
        pred_baseline[test_idx] = fold_baseline.model.predict(X_baseline)
        held_out = ", ".join(sorted(set(test["well_id"])))
        print(f"Fold {fold}: held out {held_out}")

    print("=== GAP 1 held-out-well evaluation ===")
    print(f"Wells: {len(unique_wells)} | cycles: {len(merged)} | folds: {splitter.n_splits}")
    print(f"{'Target':30} {'Linear MAE':>12} {'Boosting MAE':>14}")
    for index, target in enumerate(targets):
        base_mae = mean_absolute_error(actual[:, index], pred_baseline[:, index])
        upgraded_mae = mean_absolute_error(actual[:, index], pred_upgraded[:, index])
        print(f"{target:30} {base_mae:12.3f} {upgraded_mae:14.3f}")

    return ReservoirCycleForecaster(dataset), LinearRegressionBaseline(dataset)


def compare_optimizer_recommendations(dataset, upgraded, baseline, n_wells=3):
    """Compare seeded optimizer recommendations for representative wells."""
    from pipeline import WellDigitalTwin

    twin = WellDigitalTwin(dataset)
    latest = (dataset.css_cycles.sort_values("cycle_number")
              .groupby("well_id", as_index=False).tail(1))
    latest = latest.sort_values("cycle_sor")
    quantiles = np.linspace(0, 1, min(n_wells, len(latest)))
    selected = []
    for quantile in quantiles:
        index = int(round(quantile * (len(latest) - 1)))
        well_id = latest.iloc[index]["well_id"]
        if well_id not in selected:
            selected.append(well_id)

    print("\n=== CSS optimizer comparison ===")
    for well_id in selected:
        cycles = dataset.css_cycles[dataset.css_cycles.well_id == well_id]
        last_cycle = int(cycles.cycle_number.max())
        daily = dataset.daily_production[
            (dataset.daily_production.well_id == well_id)
            & (dataset.daily_production.cycle_number == last_cycle)
        ]
        last_day = int(daily.day_index.max())
        well_features = twin._well_features(well_id)
        rod_state = twin._rod_state_at(well_id, last_cycle, last_day)
        andrade_params = twin._andrade_params_for_well(well_id)
        baseline_result = recommend_next_css_cycle(
            baseline, well_features, twin.rod_risk_model, andrade_params, rod_state, seed=0,
        )
        upgraded_result = recommend_next_css_cycle(
            upgraded, well_features, twin.rod_risk_model, andrade_params, rod_state, seed=0,
        )
        print(f"{well_id} (latest SOR {float(cycles.iloc[-1].cycle_sor):.3f})")
        print(f"  linear:   {baseline_result['recommended_params']} | "
              f"value INR {baseline_result['predicted_economic_value_inr']:,.0f}")
        print(f"  boosting: {upgraded_result['recommended_params']} | "
              f"value INR {upgraded_result['predicted_economic_value_inr']:,.0f}")
        delta = (upgraded_result["predicted_economic_value_inr"]
                 - baseline_result["predicted_economic_value_inr"])
        print(f"  predicted value difference: INR {delta:+,.0f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help="Directory containing the pipeline CSV/JSON inputs")
    parser.add_argument("--skip-optimizer", action="store_true",
                        help="Run held-out-well metrics only")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent),
                        help="Parent directory for results/ and logs/ output folders")
    args = parser.parse_args()

    dataset = TwinDataset.load(args.data_dir)
    output_root = Path(args.output_dir).resolve()
    results_dir = output_root / "results"
    logs_dir = output_root / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    results_path = results_dir / f"evaluation_{timestamp}.txt"
    log_path = logs_dir / f"evaluation_{timestamp}.log"
    transcript = io.StringIO()

    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, transcript, log_file)):
            print("=== Evaluation run ===")
            print(f"Generated at: {datetime.now().astimezone().isoformat(timespec='seconds')}")
            print(f"Dataset directory: {Path(args.data_dir).resolve()}")
            upgraded, baseline = evaluate_forecasters(dataset)
            if not args.skip_optimizer:
                compare_optimizer_recommendations(dataset, upgraded, baseline)
            print(f"\nResults saved to: {results_path}")
            print(f"Run log saved to: {log_path}")
            results_path.write_text(transcript.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()