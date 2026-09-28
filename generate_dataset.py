"""
generate_dataset.py
====================
Synthetic-but-physically-motivated dataset generator for the Baghewala
Well-to-Surface Digital Twin prototype.

WHY THIS EXISTS
----------------
The four ML "gaps" in pipeline.py (reservoir/cycle forecaster, dynamometer
card classifier, rod-failure risk model, SRP advisor) all need training
data you almost certainly won't have on day one of a hackathon. This script
generates every table the pipeline touches, built from simple physical
relationships (Andrade viscosity-temperature law, exponential thermal
decay, viscosity-driven pump fillage, hazard-based failure sampling) so
that:
  1. the numbers are internally consistent (a well with high viscosity
     really does show worse fillage, more rod-float flags, and more
     failures downstream -- there's real signal to learn, not noise), and
  2. every parameter you'd want to calibrate against real Baghewala data
     later is isolated in one place (see CONFIG below).

THIS IS NOT REAL BAGHEWALA DATA. Treat every number as illustrative /
order-of-magnitude, calibrated loosely to public figures (17-19 deg API,
~46-48 deg C reservoir temperature, ~10,000-13,000 cP viscosity at 50 deg C,
CSS + SRP as the lift/EOR combination). Replace CONFIG and the physical
relationships with real lab/field data as it becomes available -- the
column names and shapes are what matter, since pipeline.py's model gaps
are contracted against those.

OUTPUT
------
Writes to ./baghewala_dataset/:
  well_master.csv
  pvt_viscosity_samples.csv
  css_cycles.csv
  daily_production.csv
  srp_operations.csv
  dynamometer_cards.json
  rod_failures.csv
  data_dictionary.md   <- describes every column in every file
"""

import json
import math
import os
from datetime import timedelta, date

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG -- change these to match real field parameters as you get them
# ---------------------------------------------------------------------------
SEED = 42
N_WELLS = 36
RESERVOIR_TEMP_C_RANGE = (46.0, 48.0)
API_GRAVITY_RANGE = (17.0, 19.0)
VISCOSITY_AT_RESERVOIR_TEMP_CP_RANGE = (10_000, 13_000)   # anchor point for Andrade fit
STEAM_TEMP_C = 180.0                                       # saturated steam anchor for Andrade fit
VISCOSITY_AT_STEAM_TEMP_CP_RANGE = (20, 60)                # illustrative order-of-magnitude drop
CYCLES_PER_WELL_RANGE = (4, 10)
MIN_PRODUCTION_DAYS = 30
ECONOMIC_RATE_FRACTION = 0.50
ECONOMIC_WATER_CUT = 0.48
ECONOMIC_CUTOFF_CONFIRM_DAYS = 5
MAX_PRODUCTION_DAYS = 90
SIM_START_DATE = date(2019, 1, 1)
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baghewala_dataset")

FAULT_LABELS = [
    "normal",
    "rod_float",
    "fluid_pound",
    "gas_interference",
    "worn_valve",
    "parted_rod",
]

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------
def andrade_fit(temp_c_1, visc_cp_1, temp_c_2, visc_cp_2):
    """Fit ln(eta) = A + B/T (T in Kelvin) from two anchor points.
    This is the same two-parameter form used in the heavy-oil viscosity
    literature; in the real pipeline you'd fit this from pvt_viscosity_samples
    per well instead of reading it off well_master directly."""
    t1, t2 = temp_c_1 + 273.15, temp_c_2 + 273.15
    y1, y2 = math.log(visc_cp_1), math.log(visc_cp_2)
    B = (y1 - y2) / (1 / t1 - 1 / t2)
    A = y1 - B / t1
    return A, B


def andrade_viscosity(temp_c, A, B):
    """Viscosity (cP) at a given temperature (deg C) from Andrade parameters."""
    t_kelvin = np.asarray(temp_c, dtype=float) + 273.15
    return np.exp(A + B / t_kelvin)


# ---------------------------------------------------------------------------
# 1. Well master data
# ---------------------------------------------------------------------------
def generate_well_master(n_wells=N_WELLS):
    rows = []
    for i in range(n_wells):
        well_id = f"BGW-{i + 1:02d}"
        api = rng.uniform(*API_GRAVITY_RANGE)
        reservoir_temp_c = rng.uniform(*RESERVOIR_TEMP_C_RANGE)
        visc_at_res = rng.uniform(*VISCOSITY_AT_RESERVOIR_TEMP_CP_RANGE)
        visc_at_steam = rng.uniform(*VISCOSITY_AT_STEAM_TEMP_CP_RANGE)
        A, B = andrade_fit(reservoir_temp_c, visc_at_res, STEAM_TEMP_C, visc_at_steam)
        reservoir_depth_m = rng.uniform(850, 1100)
        rows.append(dict(
            well_id=well_id,
            api_gravity=round(api, 2),
            reservoir_temp_c=round(reservoir_temp_c, 2),
            reservoir_depth_m=round(reservoir_depth_m, 1),
            pump_setting_depth_m=round(reservoir_depth_m - rng.uniform(15, 40), 1),
            rod_string_od_in=rng.choice([0.75, 0.875, 1.0]),
            rod_material=rng.choice(["steel_grade_D", "steel_grade_EL", "fiberglass_hybrid"],
                                     p=[0.55, 0.30, 0.15]),
            tubing_id_in=rng.choice([2.441, 2.992]),
            completion_year=int(rng.integers(2017, 2022)),
            initial_reservoir_pressure_kpa=round(rng.uniform(3000, 5000), 0),
            # ground-truth Andrade params used only to DRIVE the simulation below.
            # in the real pipeline these are estimated FROM pvt_viscosity_samples,
            # not read directly -- kept here for transparency/debugging only.
            _true_andrade_A=A,
            _true_andrade_B=B,
            viscosity_at_reservoir_temp_cp=round(float(andrade_viscosity(reservoir_temp_c, A, B)), 0),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. PVT / lab viscosity samples (the calibration data for the viscosity model)
# ---------------------------------------------------------------------------
def generate_pvt_viscosity_samples(well_master):
    rows = []
    sample_id = 0
    for _, w in well_master.iterrows():
        test_temps = np.sort(rng.uniform(20, 160, size=int(rng.integers(8, 15))))
        for t in test_temps:
            true_visc = andrade_viscosity(t, w["_true_andrade_A"], w["_true_andrade_B"])
            noisy_visc = true_visc * rng.lognormal(mean=0, sigma=0.06)  # ~6% lab noise
            rows.append(dict(
                sample_id=sample_id,
                well_id=w["well_id"],
                temperature_c=round(float(t), 1),
                viscosity_cp=round(float(noisy_visc), 1),
            ))
            sample_id += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3+4. CSS cycles and the daily production/thermal/viscosity time series
#      inside each cycle's production phase (built together since the
#      cycle summary columns are aggregates of the daily simulation)
# ---------------------------------------------------------------------------
def _heat_efficiency(steam_volume_bbl, soak_time_days):
    vol_term = 1 - math.exp(-steam_volume_bbl / 2500.0)
    soak_term = 1 - math.exp(-soak_time_days / 3.0)
    return 0.35 + 0.45 * vol_term * (0.6 + 0.4 * soak_term)  # bounded ~0.35-0.8


def generate_css_cycles_and_production(well_master):
    cycle_rows = []
    daily_rows = []
    for _, w in well_master.iterrows():
        well_id = w["well_id"]
        A, B = w["_true_andrade_A"], w["_true_andrade_B"]
        res_temp = w["reservoir_temp_c"]
        res_visc = w["viscosity_at_reservoir_temp_cp"]
        n_cycles = int(rng.integers(*CYCLES_PER_WELL_RANGE))
        cur_date = SIM_START_DATE + timedelta(days=int(rng.integers(0, 200)))
        base_water_cut = rng.uniform(0.15, 0.35)
        rod_float_visc_threshold = res_visc * rng.uniform(0.55, 0.75)
        max_rate_potential = rng.uniform(35, 70)  # bopd at "ideal" (post-steam) viscosity

        cum_steam_to_date = 0.0
        for cyc in range(1, n_cycles + 1):
            steam_volume = rng.uniform(2500, 6500) * (1 + 0.03 * cyc)  # mild upward drift, typical of aging wells
            injection_pressure_kpa = rng.uniform(6000, 9500)
            injection_duration_days = round(steam_volume / rng.uniform(350, 500), 1)
            soak_time_days = rng.uniform(3, 7)
            eff = _heat_efficiency(steam_volume, soak_time_days)
            peak_post_soak_temp_c = res_temp + (STEAM_TEMP_C - res_temp) * eff
            decay_rate = (rng.uniform(0.05, 0.11)) / (1 + 0.06 * soak_time_days)

            cur_date = cur_date + timedelta(days=int(injection_duration_days + soak_time_days))
            cycle_start_date = cur_date

            # --- simulate production day by day until economic cutoff ---
            cum_oil, production_days = 0.0, 0
            di = rng.uniform(0.01, 0.03)  # mild hyperbolic-ish decline rate
            cutoff_reason = "next_cycle_scheduled"
            consecutive_cutoff_days = 0
            for day in range(MAX_PRODUCTION_DAYS):
                temp_c = res_temp + (peak_post_soak_temp_c - res_temp) * math.exp(-decay_rate * day)
                temp_c += rng.normal(0, 0.3)
                visc_cp = float(andrade_viscosity(temp_c, A, B) * rng.lognormal(0, 0.04))

                visc_ratio = res_visc / max(visc_cp, 1.0)
                decline = (1 + 2.0 * di * day) ** (-1 / 2.0)
                oil_rate = max_rate_potential * (visc_ratio ** 0.4) * decline
                oil_rate *= rng.lognormal(0, 0.05)

                water_cut = min(0.95, base_water_cut + 0.003 * day + rng.normal(0, 0.01))
                liquid_rate = oil_rate / max(1 - water_cut, 0.05)
                water_rate = liquid_rate - oil_rate

                fillage_pct = 100 * np.clip(
                    1 - 0.9 * max(visc_cp / rod_float_visc_threshold - 1, 0), 0.15, 1.0
                )
                fillage_pct = float(np.clip(fillage_pct + rng.normal(0, 3), 12, 100))
                rod_float_flag = bool(visc_cp > rod_float_visc_threshold and fillage_pct < 65)

                bhp_kpa = w["initial_reservoir_pressure_kpa"] * (1 - 0.15 * (day / MAX_PRODUCTION_DAYS)) + rng.normal(0, 30)

                daily_rows.append(dict(
                    well_id=well_id, cycle_number=cyc, day_index=day,
                    date=(cycle_start_date + timedelta(days=day)).isoformat(),
                    wellhead_temp_c=round(temp_c, 2),
                    viscosity_cp=round(visc_cp, 1),
                    oil_rate_bopd=round(oil_rate, 2),
                    water_rate_bwpd=round(water_rate, 2),
                    water_cut=round(water_cut, 3),
                    bhp_kpa=round(float(bhp_kpa), 1),
                    pump_fillage_pct=round(fillage_pct, 1),
                    rod_float_flag=rod_float_flag,
                ))
                cum_oil += oil_rate
                production_days = day + 1

                below_economic_limit = (
                    day + 1 >= MIN_PRODUCTION_DAYS
                    and oil_rate < ECONOMIC_RATE_FRACTION * max_rate_potential
                    and water_cut > ECONOMIC_WATER_CUT
                )
                consecutive_cutoff_days = consecutive_cutoff_days + 1 if below_economic_limit else 0
                if consecutive_cutoff_days >= ECONOMIC_CUTOFF_CONFIRM_DAYS:
                    cutoff_reason = "rate_below_economic_limit"
                    break

            cum_steam_to_date += steam_volume
            cycle_rows.append(dict(
                well_id=well_id, cycle_number=cyc,
                cycle_start_date=cycle_start_date.isoformat(),
                steam_volume_cwe_bbl=round(steam_volume, 1),
                injection_pressure_kpa=round(injection_pressure_kpa, 1),
                injection_duration_days=injection_duration_days,
                soak_time_days=round(soak_time_days, 1),
                cumulative_steam_to_date_bbl=round(cum_steam_to_date, 1),
                peak_post_soak_temp_c=round(peak_post_soak_temp_c, 2),
                thermal_decay_rate_per_day=round(decay_rate, 4),
                production_days=production_days,
                cycle_cum_oil_bbl=round(cum_oil, 1),
                cycle_sor=round(steam_volume / max(cum_oil, 1.0), 3),
                cutoff_reason=cutoff_reason,
            ))
            cur_date = cur_date + timedelta(days=production_days + int(rng.integers(2, 10)))  # workover/idle gap

    return pd.DataFrame(cycle_rows), pd.DataFrame(daily_rows)


# ---------------------------------------------------------------------------
# 5. SRP / VFD operating time series
#    Current practice = mostly FIXED speed set manually (per the problem
#    statement), with occasional late/reactive manual step-changes. This is
#    deliberately the "before" state the new system is meant to improve on.
# ---------------------------------------------------------------------------
def generate_srp_operations(well_master, daily_production):
    rows = []
    wm = well_master.set_index("well_id")
    for (well_id, cycle), grp in daily_production.groupby(["well_id", "cycle_number"]):
        base_spm = rng.uniform(3.5, 6.5)
        base_stroke = rng.choice([86, 100, 120, 144])  # inches, common API stroke lengths
        spm = base_spm
        consecutive_float_days = 0
        manual_intervention_done = False
        for _, r in grp.sort_values("day_index").iterrows():
            if r["rod_float_flag"]:
                consecutive_float_days += 1
            else:
                consecutive_float_days = 0

            # reactive manual fix: only happens AFTER several days of float,
            # and only sometimes -- this lag is exactly the gap the SRP
            # advisor (pipeline.py gap 4) is meant to close.
            if (not manual_intervention_done and consecutive_float_days >= 6
                    and rng.random() < 0.5):
                spm = max(spm - rng.uniform(0.8, 1.5), 2.0)
                manual_intervention_done = True

            vfd_freq_hz = spm * rng.uniform(9.5, 10.5)
            load_factor = (r["viscosity_cp"] / wm.loc[well_id, "viscosity_at_reservoir_temp_cp"])
            motor_current_a = 28 + 14 * min(load_factor, 1.5) + rng.normal(0, 1.5)
            if r["rod_float_flag"]:
                motor_current_a *= 0.85  # rods under-loaded during the float phase itself
            impact_load_proxy = 0.15 + (2.2 if r["rod_float_flag"] else 0.0) + rng.normal(0, 0.1)

            rows.append(dict(
                well_id=well_id, cycle_number=cycle, day_index=r["day_index"],
                spm=round(spm, 2),
                stroke_length_in=base_stroke,
                vfd_freq_hz=round(vfd_freq_hz, 2),
                motor_current_a=round(max(motor_current_a, 5), 2),
                motor_power_kw=round(max(motor_current_a, 5) * 0.38 * rng.uniform(0.95, 1.05), 2),
                impact_load_proxy=round(max(impact_load_proxy, 0), 3),
                manual_spm_intervention=bool(manual_intervention_done and consecutive_float_days == 0),
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Synthetic dynamometer cards
#    Simplified PARAMETRIC stand-in for a true wave-equation card (see
#    libzrod for a real Gibbs-method engine). Good enough to be visually
#    and statistically distinguishable per fault class for a first
#    classifier; replace with wave-equation-simulated cards for higher
#    fidelity once you have real rod-string parameters.
# ---------------------------------------------------------------------------
def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def synth_card(fault_label, l_min, l_max, n_points=60, noise=0.02, rng_local=None):
    rng_local = rng_local or rng
    half = n_points // 2
    x_up = np.linspace(0, 1, half)
    x_down = np.linspace(1, 0, n_points - half)

    load_up = l_min + (l_max - l_min) * _smoothstep(x_up)

    if fault_label == "normal":
        load_down = l_min + (l_max - l_min) * (_smoothstep(x_down) ** 1.3)
    elif fault_label == "rod_float":
        # load stays high late into the downstroke (delayed rod fall), then
        # drops sharply -- the "bunched" transfer signature described in
        # the rod-float literature.
        transition = 0.25
        load_down = np.where(
            x_down > transition,
            l_max - (l_max - l_min) * 0.08 * (1 - x_down),
            l_min + (l_max - l_min) * (x_down / transition) ** 0.5,
        )
    elif fault_label == "fluid_pound":
        # sudden load drop (impact) at a specific point near bottom of stroke
        pound_point = 0.15
        load_down = l_min + (l_max - l_min) * _smoothstep(x_down)
        drop_mask = x_down < pound_point
        load_down = np.where(drop_mask, l_min + (l_max - l_min) * 0.05, load_down)
    elif fault_label == "gas_interference":
        compress = 0.35  # card area shrinks
        mid = (l_max + l_min) / 2
        load_up = mid + (load_up - mid) * (1 - compress)
        load_down = mid + (l_min + (l_max - l_min) * _smoothstep(x_down) - mid) * (1 - compress)
    elif fault_label == "worn_valve":
        # leaky valve -> sloped sides instead of flat top/bottom, area shrinks gradually
        load_up = l_min + (l_max - l_min) * (0.15 * x_up + 0.85 * _smoothstep(x_up))
        load_down = l_min + (l_max - l_min) * (0.15 * (1 - x_down) + 0.85 * _smoothstep(x_down) * 0.75)
    elif fault_label == "parted_rod":
        load_up = np.full_like(x_up, l_min) + (l_max - l_min) * 0.05
        load_down = np.full_like(x_down, l_min)
    else:
        raise ValueError(f"unknown fault_label: {fault_label}")

    position = np.concatenate([x_up, x_down])
    load = np.concatenate([load_up, load_down])
    load = load * (1 + rng_local.normal(0, noise, size=load.shape))
    return position.tolist(), load.tolist()


def generate_dynamometer_cards(well_master, srp_operations, daily_production):
    wm = well_master.set_index("well_id")
    daily_idx = daily_production.set_index(["well_id", "cycle_number", "day_index"])
    cards = []
    card_id = 0
    for (well_id, cycle), grp in srp_operations.groupby(["well_id", "cycle_number"]):
        depth_m = wm.loc[well_id, "pump_setting_depth_m"]
        rod_weight_load_lbf = depth_m * 3.28084 * rng.uniform(0.9, 1.1)  # crude proxy for dead rod weight
        sample_days = sorted(rng.choice(grp["day_index"].values,
                                         size=min(3, len(grp)), replace=False))
        for day in sample_days:
            row = grp[grp["day_index"] == day].iloc[0]
            daily = daily_idx.loc[(well_id, cycle, day)]
            l_min = rod_weight_load_lbf * rng.uniform(0.85, 1.0)
            l_max = l_min + rod_weight_load_lbf * rng.uniform(0.6, 1.0) * (daily["pump_fillage_pct"] / 100)

            if daily["rod_float_flag"]:
                label = "rod_float"
            else:
                # low base-rate occurrence of other faults for class diversity
                label = rng.choice(
                    ["normal", "fluid_pound", "gas_interference", "worn_valve", "parted_rod"],
                    p=[0.80, 0.06, 0.06, 0.06, 0.02],
                )
            position, load = synth_card(label, l_min, l_max)
            cards.append(dict(
                card_id=card_id, well_id=well_id, cycle_number=int(cycle),
                day_index=int(day), fault_label=label,
                spm=float(row["spm"]), motor_current_a=float(row["motor_current_a"]),
                position=position, load=load,
            ))
            card_id += 1
    return cards


# ---------------------------------------------------------------------------
# 7. Rod failure history (hazard driven by cumulative rod-float exposure)
# ---------------------------------------------------------------------------
def generate_rod_failures(well_master, srp_operations, css_cycles):
    rows = []
    failure_id = 0
    failure_types = ["parted_rod", "pump_wear", "tubing_leak", "coupling_failure"]
    wm = well_master.set_index("well_id")
    for well_id, grp in srp_operations.groupby("well_id"):
        grp = grp.sort_values(["cycle_number", "day_index"])
        cum_float_days = 0
        age_days = 0
        n_failures = 0
        for _, r in grp.iterrows():
            age_days += 1
            if r["impact_load_proxy"] > 1.0:
                cum_float_days += 1
            # simple logistic hazard: baseline age risk + strong float-exposure term
            hazard = 1 / (1 + math.exp(-(-9.5 + 0.0006 * age_days + 0.02 * cum_float_days)))
            if rng.random() < hazard and n_failures < 3:
                depth = wm.loc[well_id, "pump_setting_depth_m"]
                rows.append(dict(
                    failure_id=failure_id, well_id=well_id,
                    cycle_number_at_failure=int(r["cycle_number"]),
                    day_index_at_failure=int(r["day_index"]),
                    rod_age_days=age_days,
                    cumulative_rod_float_days=cum_float_days,
                    failure_type=rng.choice(failure_types, p=[0.45, 0.30, 0.15, 0.10]),
                    depth_of_failure_m=round(depth * rng.uniform(0.7, 1.0), 1),
                ))
                failure_id += 1
                n_failures += 1
                cum_float_days = 0  # rod string replaced, exposure resets
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data dictionary
# ---------------------------------------------------------------------------
DATA_DICTIONARY = """\
# Baghewala synthetic dataset -- data dictionary

All files are illustrative/synthetic (see generate_dataset.py docstring).
Column units are noted in brackets.

## well_master.csv
One row per well.
- well_id: unique well identifier
- api_gravity [deg API]
- reservoir_temp_c [deg C]: static reservoir temperature
- reservoir_depth_m, pump_setting_depth_m [m]
- rod_string_od_in [in], rod_material, tubing_id_in [in]
- completion_year
- initial_reservoir_pressure_kpa [kPa]
- viscosity_at_reservoir_temp_cp [cP]: live-oil viscosity at reservoir_temp_c

## pvt_viscosity_samples.csv
Lab-style viscosity-vs-temperature points per well -- the calibration data
for a per-well Andrade (or other heavy-oil) viscosity-temperature model.
- well_id, temperature_c [deg C], viscosity_cp [cP]

## css_cycles.csv
One row per CSS cycle per well -- the cycle-level record the reservoir/
cycle forecaster (pipeline.py Gap 1) is trained against.
- well_id, cycle_number, cycle_start_date
- steam_volume_cwe_bbl [bbl cold-water-equivalent]
- injection_pressure_kpa [kPa], injection_duration_days, soak_time_days
- cumulative_steam_to_date_bbl [bbl]
- peak_post_soak_temp_c [deg C]: near-wellbore temperature right after soak
- thermal_decay_rate_per_day: rate constant for T(t) = T_res + (T_peak-T_res)*exp(-rate*t)
- production_days: length of the production phase before cutoff
- cycle_cum_oil_bbl [bbl], cycle_sor [bbl steam / bbl oil]
- cutoff_reason

## daily_production.csv
Day-by-day record within each cycle's production phase.
- well_id, cycle_number, day_index, date
- wellhead_temp_c [deg C], viscosity_cp [cP]
- oil_rate_bopd, water_rate_bwpd [bbl/day], water_cut [fraction]
- bhp_kpa [kPa]
- pump_fillage_pct [%]
- rod_float_flag: ground-truth label for whether the pump was floating that day

## srp_operations.csv
Day-by-day SRP/VFD telemetry, joined to daily_production on
(well_id, cycle_number, day_index).
- spm [strokes/min], stroke_length_in [in], vfd_freq_hz [Hz]
- motor_current_a [A], motor_power_kw [kW]
- impact_load_proxy: unitless severity score for downstroke impact loading
- manual_spm_intervention: True on the day a reactive manual SPM cut happened

## dynamometer_cards.json
List of synthetic surface dynamometer cards -- the training data for the
card classifier (pipeline.py Gap 2).
- card_id, well_id, cycle_number, day_index
- fault_label: one of normal / rod_float / fluid_pound / gas_interference /
  worn_valve / parted_rod
- spm, motor_current_a: matching SRP telemetry at that timestamp
- position: list[float], 0->1->0 over one stroke (dimensionless, fraction of stroke)
- load: list[float] [lbf], same length as position

## rod_failures.csv
Failure events -- the training data for the rod-failure risk model
(pipeline.py Gap 3).
- failure_id, well_id, cycle_number_at_failure, day_index_at_failure
- rod_age_days: days since last replacement/completion at time of failure
- cumulative_rod_float_days: rod-float-flagged days accumulated since last replacement
- failure_type: parted_rod / pump_wear / tubing_leak / coupling_failure
- depth_of_failure_m [m]
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    well_master = generate_well_master()
    pvt = generate_pvt_viscosity_samples(well_master)
    css_cycles, daily_production = generate_css_cycles_and_production(well_master)
    srp_operations = generate_srp_operations(well_master, daily_production)
    cards = generate_dynamometer_cards(well_master, srp_operations, daily_production)
    rod_failures = generate_rod_failures(well_master, srp_operations, css_cycles)

    # drop the "cheat" columns before saving well_master (they were only
    # needed to drive the simulation -- a real classifier/forecaster should
    # never see them, since in reality they don't exist until you fit them)
    well_master_public = well_master.drop(columns=["_true_andrade_A", "_true_andrade_B"])

    well_master_public.to_csv(os.path.join(OUTDIR, "well_master.csv"), index=False)
    pvt.to_csv(os.path.join(OUTDIR, "pvt_viscosity_samples.csv"), index=False)
    css_cycles.to_csv(os.path.join(OUTDIR, "css_cycles.csv"), index=False)
    daily_production.to_csv(os.path.join(OUTDIR, "daily_production.csv"), index=False)
    srp_operations.to_csv(os.path.join(OUTDIR, "srp_operations.csv"), index=False)
    rod_failures.to_csv(os.path.join(OUTDIR, "rod_failures.csv"), index=False)
    with open(os.path.join(OUTDIR, "dynamometer_cards.json"), "w") as f:
        json.dump(cards, f)
    with open(os.path.join(OUTDIR, "data_dictionary.md"), "w") as f:
        f.write(DATA_DICTIONARY)

    print(f"Wrote dataset to {OUTDIR}/")
    print(f"  well_master:          {len(well_master_public):>6} rows")
    print(f"  pvt_viscosity_samples:{len(pvt):>6} rows")
    print(f"  css_cycles:           {len(css_cycles):>6} rows")
    print(f"  daily_production:     {len(daily_production):>6} rows")
    print(f"  srp_operations:       {len(srp_operations):>6} rows")
    print(f"  dynamometer_cards:    {len(cards):>6} cards"
          f"  (labels: {pd.Series([c['fault_label'] for c in cards]).value_counts().to_dict()})")
    print(f"  rod_failures:         {len(rod_failures):>6} rows")


if __name__ == "__main__":
    main()
