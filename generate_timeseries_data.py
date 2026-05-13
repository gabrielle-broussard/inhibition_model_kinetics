"""
generate_timeseries_data.py

Generate optional batch time-series CSV datasets for the inhibition-kinetics project.

This script uses the same batch biomass/substrate/product ODE structure for all
growth models while swapping in each model's literature-derived specific growth
rate expression, mu(S).

State equations:
    dX/dt = mu(S) X
    dS/dt = -(1/Yxs) mu(S) X
    dP/dt = Ypx mu(S) X

CSV outputs:
    <output_dir>/csv/timeseries_model_metadata.csv
    <output_dir>/csv/timeseries_profiles_all_models.csv
    <output_dir>/csv/initial_substrate_loading_sweep_summary.csv
    <output_dir>/csv/monod_overprediction_summary.csv

"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


EPS = 1e-12
MODEL_ORDER = [
    "Monod",
    "Haldane-Andrews",
    "Aiba-Edwards",
    "Luong",
    "Han-Levenspiel",
]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sort_models(models: Iterable[str]) -> List[str]:
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda m: order.get(m, 999))


def load_model_module(script_path: str | Path):

    script_path = Path(script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find model script: {script_path}")

    spec = importlib.util.spec_from_file_location("model_library", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import model script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["model_library"] = module
    spec.loader.exec_module(module)

    for required in ["MODELS", "term_decomposition", "load_literature_true_cases"]:
        if not hasattr(module, required):
            raise AttributeError(f"{script_path.name} is missing required object: {required}")

    return module


def row_value(row: pd.Series, col: str, default=np.nan):
    return row[col] if col in row.index else default


def as_float_or_nan(value) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


# -----------------------------------------------------------------------------
# Batch ODE model
# -----------------------------------------------------------------------------
def evaluate_mu(model_module, model_name: str, S: float, params: Sequence[float]) -> float:

    S_eff = max(float(S), 0.0)
    mu = model_module.MODELS[model_name]["func"](np.array([S_eff], dtype=float), *params)
    return float(np.asarray(mu).ravel()[0])


def batch_rhs(t: float, y: np.ndarray, model_module, model_name: str, params: Sequence[float], Yxs: float, Ypx: float) -> np.ndarray:

    X, S, P = y
    X = max(float(X), 0.0)
    S = max(float(S), 0.0)

    mu = evaluate_mu(model_module, model_name, S, params)

    dXdt = mu * X
    dSdt = -(1.0 / max(Yxs, EPS)) * mu * X
    dPdt = Ypx * mu * X
    return np.array([dXdt, dSdt, dPdt], dtype=float)


def substrate_depleted_event(t, y, *args):

    return y[1] - 1e-9


substrate_depleted_event.terminal = True
substrate_depleted_event.direction = -1


def simulate_one_profile(
    model_module,
    model_name: str,
    params: Sequence[float],
    X0: float,
    S0: float,
    P0: float,
    Yxs: float,
    Ypx: float,
    t_end: float,
    dt: float,
    stop_at_depletion: bool = False,
) -> pd.DataFrame:

    t_eval = np.arange(0.0, t_end + dt, dt)
    events = substrate_depleted_event if stop_at_depletion else None

    sol = solve_ivp(
        fun=lambda t, y: batch_rhs(t, y, model_module, model_name, params, Yxs, Ypx),
        t_span=(0.0, t_end),
        y0=np.array([X0, S0, P0], dtype=float),
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-7,
        atol=1e-10,
        events=events,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solve failed for {model_name}, S0={S0}: {sol.message}")

    X = np.maximum(sol.y[0], 0.0)
    S = np.maximum(sol.y[1], 0.0)
    P = np.maximum(sol.y[2], 0.0)

    mu = np.array([evaluate_mu(model_module, model_name, s, params) for s in S], dtype=float)
    specific_product_rate = Ypx * mu * X
    specific_substrate_rate = (1.0 / max(Yxs, EPS)) * mu * X

    df = pd.DataFrame(
        {
            "time": sol.t,
            "model": model_name,
            "X_biomass": X,
            "S_substrate": S,
            "P_product": P,
            "mu": mu,
            "X_over_S": X / (S + EPS),
            "dPdt": specific_product_rate,
            "substrate_consumption_rate": specific_substrate_rate,
        }
    )
    return df


def add_structural_terms(model_module, profiles: pd.DataFrame, params_by_model: Mapping[str, Sequence[float]]) -> pd.DataFrame:

    frames = []
    for model_name, group in profiles.groupby("model", sort=False):
        group = group.copy()
        params = params_by_model[model_name]
        S = group["S_substrate"].to_numpy(dtype=float)
        terms = model_module.term_decomposition(model_name, S, params)
        for term_name, values in terms.items():
            group[f"term_{term_name}"] = np.asarray(values, dtype=float)
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


# -----------------------------------------------------------------------------
# Model and literature loading
# -----------------------------------------------------------------------------
def load_cases(model_module, parameter_library: str | Path):

    cases, primary = model_module.load_literature_true_cases(str(parameter_library))
    primary = primary.copy()
    primary["model"] = primary["model"].astype(str)
    return cases, primary


def build_model_metadata(cases: Mapping[str, dict], primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in sort_models(cases.keys()):
        row = primary[primary["model"] == model_name].iloc[0]
        item = {
            "model": model_name,
            "parameter_set_id": row_value(row, "parameter_set_id", ""),
            "source_short": row_value(row, "source_short", ""),
            "source_title": row_value(row, "source_title", ""),
            "organism": row_value(row, "organism", ""),
            "substrate": row_value(row, "substrate", ""),
            "reactor_mode": row_value(row, "reactor_mode", ""),
            "S_min": row_value(row, "S_min", np.nan),
            "S_max": row_value(row, "S_max", np.nan),
            "S_units": row_value(row, "S_units", ""),
        }
        rows.append(item)
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Sweep calculations
# -----------------------------------------------------------------------------
def summarize_final_state(profile: pd.DataFrame) -> dict:
    final = profile.sort_values("time").iloc[-1]
    initial = profile.sort_values("time").iloc[0]
    S0 = float(initial["S_substrate"])
    P0 = float(initial["P_product"])
    X0 = float(initial["X_biomass"])
    final_S = float(final["S_substrate"])
    final_P = float(final["P_product"])
    final_X = float(final["X_biomass"])
    consumed_S = max(S0 - final_S, 0.0)

    return {
        "final_time": float(final["time"]),
        "initial_biomass": X0,
        "initial_substrate": S0,
        "initial_product": P0,
        "final_biomass": final_X,
        "final_substrate": final_S,
        "final_product": final_P,
        "substrate_consumed": consumed_S,
        "apparent_product_per_initial_substrate": (final_P - P0) / max(S0, EPS),
        "apparent_product_per_consumed_substrate": (final_P - P0) / max(consumed_S, EPS),
        "apparent_biomass_per_consumed_substrate": (final_X - X0) / max(consumed_S, EPS),
    }


def run_s0_sweep(
    model_module,
    cases: Mapping[str, dict],
    X0: float,
    P0: float,
    Yxs: float,
    Ypx: float,
    t_end: float,
    dt: float,
    s0_fractions: Sequence[float],
    stop_at_depletion: bool,
) -> pd.DataFrame:
    rows = []
    for model_name in sort_models(cases.keys()):
        params = cases[model_name]["params"]
        Smax = float(cases[model_name]["s_max"])
        S_units = cases[model_name].get("S_units", "")
        for frac in s0_fractions:
            S0 = float(frac) * Smax
            profile = simulate_one_profile(
                model_module=model_module,
                model_name=model_name,
                params=params,
                X0=X0,
                S0=S0,
                P0=P0,
                Yxs=Yxs,
                Ypx=Ypx,
                t_end=t_end,
                dt=dt,
                stop_at_depletion=stop_at_depletion,
            )
            summary = summarize_final_state(profile)
            summary.update(
                {
                    "model": model_name,
                    "S0_fraction_of_Smax": float(frac),
                    "Smax_literature": Smax,
                    "S_units": S_units,
                }
            )
            rows.append(summary)
    return pd.DataFrame(rows)


def run_monod_overprediction_sweep(
    model_module,
    cases: Mapping[str, dict],
    X0: float,
    P0: float,
    Yxs: float,
    Ypx: float,
    t_end: float,
    dt: float,
    s0_fractions: Sequence[float],
    stop_at_depletion: bool,
) -> pd.DataFrame:

    if "Monod" not in model_module.MODELS:
        return pd.DataFrame()

    rows = []
    for model_name in sort_models(cases.keys()):
        if model_name == "Monod":
            continue

        params = cases[model_name]["params"]
        param_names = model_module.MODELS[model_name]["params"]
        param_map = dict(zip(param_names, params))
        if "mumax" not in param_map or "Ks" not in param_map:
            continue

        monod_like_params = [float(param_map["mumax"]), float(param_map["Ks"])]
        Smax = float(cases[model_name]["s_max"])
        S_units = cases[model_name].get("S_units", "")

        for frac in s0_fractions:
            S0 = float(frac) * Smax

            true_profile = simulate_one_profile(
                model_module=model_module,
                model_name=model_name,
                params=params,
                X0=X0,
                S0=S0,
                P0=P0,
                Yxs=Yxs,
                Ypx=Ypx,
                t_end=t_end,
                dt=dt,
                stop_at_depletion=stop_at_depletion,
            )
            monod_profile = simulate_one_profile(
                model_module=model_module,
                model_name="Monod",
                params=monod_like_params,
                X0=X0,
                S0=S0,
                P0=P0,
                Yxs=Yxs,
                Ypx=Ypx,
                t_end=t_end,
                dt=dt,
                stop_at_depletion=stop_at_depletion,
            )

            true_final = summarize_final_state(true_profile)
            monod_final = summarize_final_state(monod_profile)

            P_true = true_final["final_product"]
            P_monod = monod_final["final_product"]
            overprediction = P_monod - P_true
            overprediction_pct = 100.0 * overprediction / max(abs(P_true), EPS)

            rows.append(
                {
                    "inhibited_model": model_name,
                    "comparison_model": "Monod using same mumax and Ks",
                    "S0_fraction_of_Smax": float(frac),
                    "S0": S0,
                    "S_units": S_units,
                    "final_product_inhibited": P_true,
                    "final_product_monod_comparison": P_monod,
                    "product_overprediction": overprediction,
                    "product_overprediction_percent": overprediction_pct,
                    "final_biomass_inhibited": true_final["final_biomass"],
                    "final_biomass_monod_comparison": monod_final["final_biomass"],
                    "final_substrate_inhibited": true_final["final_substrate"],
                    "final_substrate_monod_comparison": monod_final["final_substrate"],
                }
            )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate all-model dynamic time-series CSV data for Problem 8.")
    parser.add_argument("--model_script", default="generate_model_recovery_data.py")
    parser.add_argument("--parameter_library", default="literature_parameter_sets.csv")
    parser.add_argument("--output_dir", default="timeseries_all_models_data")
    parser.add_argument("--X0", type=float, default=0.05, help="Initial biomass concentration in arbitrary consistent units.")
    parser.add_argument("--P0", type=float, default=0.0, help="Initial product concentration.")
    parser.add_argument("--Yxs", type=float, default=0.5, help="Biomass yield on substrate.")
    parser.add_argument("--Ypx", type=float, default=0.2, help="Product yield on biomass growth rate.")
    parser.add_argument("--t_end", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument(
        "--profile_s0_fraction",
        type=float,
        default=0.75,
        help="S0/Smax used for the main time-series overlay figures.",
    )
    parser.add_argument(
        "--s0_fractions",
        nargs="*",
        type=float,
        default=[0.10, 0.25, 0.50, 0.75, 1.00],
        help="Normalized S0/Smax values for the loading sweep.",
    )
    parser.add_argument(
        "--stop_at_depletion",
        action="store_true",
        help="Stop each ODE solve when substrate reaches approximately zero.",
    )
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    csv_dir = ensure_dir(output_dir / "csv")

    model_module = load_model_module(args.model_script)
    cases, primary = load_cases(model_module, args.parameter_library)
    metadata = build_model_metadata(cases, primary)
    metadata.to_csv(csv_dir / "timeseries_model_metadata.csv", index=False)

    # Main profile set: one normalized S0 per model.
    profile_frames = []
    params_by_model: Dict[str, Sequence[float]] = {}
    for model_name in sort_models(cases.keys()):
        params = cases[model_name]["params"]
        params_by_model[model_name] = params
        Smax = float(cases[model_name]["s_max"])
        S0 = float(args.profile_s0_fraction) * Smax
        profile = simulate_one_profile(
            model_module=model_module,
            model_name=model_name,
            params=params,
            X0=args.X0,
            S0=S0,
            P0=args.P0,
            Yxs=args.Yxs,
            Ypx=args.Ypx,
            t_end=args.t_end,
            dt=args.dt,
            stop_at_depletion=args.stop_at_depletion,
        )
        profile["S0"] = S0
        profile["S0_fraction_of_Smax"] = float(args.profile_s0_fraction)
        profile["Smax_literature"] = Smax
        profile["S_units"] = cases[model_name].get("S_units", "")
        profile_frames.append(profile)

    profiles = pd.concat(profile_frames, ignore_index=True)
    profiles = add_structural_terms(model_module, profiles, params_by_model)
    profiles.to_csv(csv_dir / "timeseries_profiles_all_models.csv", index=False)


    # Loading sweep.
    sweep = run_s0_sweep(
        model_module=model_module,
        cases=cases,
        X0=args.X0,
        P0=args.P0,
        Yxs=args.Yxs,
        Ypx=args.Ypx,
        t_end=args.t_end,
        dt=args.dt,
        s0_fractions=args.s0_fractions,
        stop_at_depletion=args.stop_at_depletion,
    )
    sweep.to_csv(csv_dir / "initial_substrate_loading_sweep_summary.csv", index=False)


    # Monod overprediction diagnostic.
    overprediction = run_monod_overprediction_sweep(
        model_module=model_module,
        cases=cases,
        X0=args.X0,
        P0=args.P0,
        Yxs=args.Yxs,
        Ypx=args.Ypx,
        t_end=args.t_end,
        dt=args.dt,
        s0_fractions=args.s0_fractions,
        stop_at_depletion=args.stop_at_depletion,
    )
    overprediction.to_csv(csv_dir / "monod_overprediction_summary.csv", index=False)

    readme = f"""
Main assumptions:
- Batch balances for all models:
    dX/dt = mu(S) X
    dS/dt = -(1/Yxs) mu(S) X
    dP/dt = Ypx mu(S) X
- Yxs = {args.Yxs}
- Ypx = {args.Ypx}
- X0 = {args.X0}
- P0 = {args.P0}
- t_end = {args.t_end}
- dt = {args.dt}
- Main profile substrate loading = S0/Smax = {args.profile_s0_fraction}
- Loading sweep fractions = {args.s0_fractions}

Key files:
- csv/timeseries_profiles_all_models.csv
- csv/initial_substrate_loading_sweep_summary.csv
- csv/monod_overprediction_summary.csv
- profile_plots/all_models_X_S_P_mu_timeseries.png
- profile_plots/all_models_biomass_to_substrate_ratio.png
- structural_term_plots/structural_terms_<model>.png
- final_product_vs_normalized_initial_substrate.png
- apparent_product_per_initial_substrate.png
- monod_product_overprediction_vs_loading.png

"""
    (output_dir / "README_timeseries_all_models.txt").write_text(readme.strip() + "\n", encoding="utf-8")

    print("Finished Problem 8 time-series data workflow.")
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
