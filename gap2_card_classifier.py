"""
gap2_card_classifier.py
========================
Solves GAP 2 (DynamometerCardClassifier) from pipeline.py.

WHAT'S WRONG WITH TRAINING ON THE RAW 378 CARDS AS-IS
-------------------------------------------------------
Label counts: normal 245, rod_float 77, gas_interference 22, fluid_pound 20,
worn_valve 9, parted_rod 5 -- and parted_rod appears in only 5 of 18 wells
(worn_valve in 8), so well-grouped CV would leave many folds with zero
examples of the rarest classes. This script therefore evaluates with
row-level stratified k-fold, NOT well-grouped -- a deliberate, disclosed
trade-off (see the caveat printed at the end), not an oversight.

TWO CHANGES FROM THE PLACEHOLDER
----------------------------------
1. AUGMENTATION: generate_dataset.py's own synth_card() is reused to add
   synthetic parted_rod / worn_valve / fluid_pound / gas_interference cards
   up to a workable count. Augmentation happens INSIDE each training fold
   only -- the test fold in every split is 100% real cards, so the reported
   metrics are never inflated by evaluating on synthetic data.
2. RICHER FEATURES: the placeholder's 5 hand-crafted summary features are
   kept, but a coarse resampled load curve (10 points on the upstroke, 10 on
   the downstroke) plus spm/motor_current_a are added, so the model sees
   more of the actual card shape instead of only 5 summary numbers -- the
   sklearn-only stand-in for "let a 1D-CNN see the raw sequence" (this
   sandbox has no torch/tensorflow -- see the note at the bottom of this
   file if you have a GPU environment to actually train a CNN).

USAGE
-----
    python gap2_card_classifier.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import TwinDataset, FAULT_LABELS, _extract_card_features
from generate_dataset import synth_card

RARE_TARGET_COUNT = 60  # bring each rare class up to roughly this many in TRAIN only
N_RESAMPLE_PTS = 10      # per stroke direction -> 20 shape features total


# =============================================================================
# RICHER FEATURE EXTRACTION (superset of the placeholder's 5 features)
# =============================================================================
def extract_features(card: dict) -> list:
    base = _extract_card_features(card["position"], card["load"])
    position = np.asarray(card["position"])
    load = np.asarray(card["load"])
    half = len(position) // 2
    l_min, l_max = load.min(), load.max()
    rng_ = max(l_max - l_min, 1e-6)

    up_grid = np.linspace(0, 1, N_RESAMPLE_PTS)
    down_grid = np.linspace(1, 0, N_RESAMPLE_PTS)
    up_shape = (np.interp(up_grid, position[:half], load[:half]) - l_min) / rng_
    down_shape = (np.interp(down_grid[::-1], position[half:][::-1], load[half:][::-1]) - l_min) / rng_

    spm = card.get("spm", np.nan)
    motor_current = card.get("motor_current_a", np.nan)
    return base + list(up_shape) + list(down_shape) + [spm, motor_current]


FEATURE_NAMES = (
    ["card_range", "area_ratio", "load_asymmetry", "steepest_drop_frac", "down_std_ratio"]
    + [f"up_{i}" for i in range(N_RESAMPLE_PTS)]
    + [f"down_{i}" for i in range(N_RESAMPLE_PTS)]
    + ["spm", "motor_current_a"]
)


# =============================================================================
# AUGMENTATION -- reuses generate_dataset.py's own card generator
# =============================================================================
def augment_rare_classes(train_cards: list, well_master: pd.DataFrame,
                          target_count: int = RARE_TARGET_COUNT, seed: int = 0) -> list:
    """Adds synthetic cards for under-represented classes, built from the
    SAME synth_card() parametric model used to build the real dataset, with
    l_min/l_max drawn the same way generate_dynamometer_cards() does (scaled
    off a random well's pump depth) so augmented cards sit in a realistic
    load range rather than an arbitrary one."""
    rng_local = np.random.default_rng(seed)
    by_label = {lbl: [c for c in train_cards if c["fault_label"] == lbl] for lbl in FAULT_LABELS}
    depths = well_master["pump_setting_depth_m"].values
    synthetic = []
    for label, examples in by_label.items():
        n_needed = target_count - len(examples)
        if n_needed <= 0:
            continue
        for _ in range(n_needed):
            depth_m = float(rng_local.choice(depths))
            rod_weight = depth_m * 3.28084 * rng_local.uniform(0.9, 1.1)
            fillage = rng_local.uniform(25, 95)
            l_min = rod_weight * rng_local.uniform(0.85, 1.0)
            l_max = l_min + rod_weight * rng_local.uniform(0.6, 1.0) * (fillage / 100)
            noise = rng_local.uniform(0.015, 0.04)
            position, load = synth_card(label, l_min, l_max, noise=noise, rng_local=rng_local)
            synthetic.append({
                "fault_label": label, "position": position, "load": load,
                "spm": float(rng_local.uniform(3.5, 6.5)),
                "motor_current_a": float(rng_local.uniform(20, 55)),
            })
    return train_cards + synthetic


# =============================================================================
# GAP 2 REPLACEMENT -- same .predict(card) contract as the placeholder
# =============================================================================
class DynamometerCardClassifier:
    def __init__(self, cards: list, well_master: pd.DataFrame, augment: bool = True):
        train_cards = augment_rare_classes(cards, well_master) if augment else cards
        X = [extract_features(c) for c in train_cards]
        y = [c["fault_label"] for c in train_cards]
        self._model = RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=0
        ).fit(X, y)
        self._classes = list(self._model.classes_)

    def predict(self, card: dict) -> dict:
        feats = [extract_features(card)]
        probs = self._model.predict_proba(feats)[0]
        class_probabilities = {c: float(p) for c, p in zip(self._classes, probs)}
        for label in FAULT_LABELS:
            class_probabilities.setdefault(label, 0.0)
        best_label = max(class_probabilities, key=class_probabilities.get)
        return {"predicted_label": best_label, "confidence": class_probabilities[best_label],
                "class_probabilities": class_probabilities}


# =============================================================================
# EVALUATION -- stratified row-level k-fold, baseline vs new, same folds
# =============================================================================
def evaluate(cards: list, well_master: pd.DataFrame, n_splits: int = 5):
    labels = np.array([c["fault_label"] for c in cards])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    baseline_preds, new_preds, y_true = [], [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(cards)), labels)):
        train_cards = [cards[i] for i in train_idx]
        test_cards = [cards[i] for i in test_idx]

        # --- baseline: pipeline.py's placeholder, verbatim ---
        Xb = [_extract_card_features(c["position"], c["load"]) for c in train_cards]
        yb = [c["fault_label"] for c in train_cards]
        base_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0).fit(Xb, yb)

        # --- new: enriched features + augmentation (train fold only!) ---
        new_model = DynamometerCardClassifier(train_cards, well_master, augment=True)

        for c in test_cards:
            y_true.append(c["fault_label"])
            baseline_preds.append(base_model.predict([_extract_card_features(c["position"], c["load"])])[0])
            new_preds.append(new_model.predict(c)["predicted_label"])

    print("=== BASELINE (5 features, RandomForest, no class weighting, no augmentation) ===")
    print(classification_report(y_true, baseline_preds, labels=FAULT_LABELS, zero_division=0))
    print(f"Macro-F1: {f1_score(y_true, baseline_preds, labels=FAULT_LABELS, average='macro', zero_division=0):.3f}\n")

    print("=== NEW (enriched features + spm/motor_current + augmentation + class_weight=balanced) ===")
    print(classification_report(y_true, new_preds, labels=FAULT_LABELS, zero_division=0))
    print(f"Macro-F1: {f1_score(y_true, new_preds, labels=FAULT_LABELS, average='macro', zero_division=0):.3f}")


if __name__ == "__main__":
    dataset = TwinDataset.load()
    print(f"Cards: {len(dataset.cards)} total, label counts:",
          pd.Series([c['fault_label'] for c in dataset.cards]).value_counts().to_dict(), "\n")

    evaluate(dataset.cards, dataset.well_master)

    print("\nCaveat: evaluated with ROW-level stratified k-fold, not well-grouped -- "
          "parted_rod exists in only 5 of 18 wells (worn_valve in 8), so well-grouped CV "
          "would leave folds with zero rare-class examples. This means the model may be "
          "partly learning well-specific load *scale* rather than pure fault *shape* -- "
          "worth re-checking once more wells have rare-fault cards.")

    print("\nRefitting final deployable model on ALL real cards + augmentation...")
    final_model = DynamometerCardClassifier(dataset.cards, dataset.well_master, augment=True)
    sample = dataset.cards[0]
    print("Sample prediction:", final_model.predict(sample), "| true label:", sample["fault_label"])
