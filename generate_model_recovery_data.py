"""
generate_model_recovery_data.py

Data-generation script for the five-model substrate-inhibition recovery study.

This script generates synthetic substrate-response datasets, adds documented
Gaussian noise levels across documented random seeds, fits all candidate growth
models, and writes CSV outputs.

Default statistical design:
    SEED_VALUES = 42 through 71
    N_REPLICATES_PER_SEED = 1

Outputs:
outputs/<TrueModel>/
    all_fits.csv
    best_by_replicate.csv
    summary_by_noise.csv
    summary_by_seed_noise.csv
    summary_with_ci.csv
    substrate_grid.csv
    pointwise_predictions.csv
    study_metadata.csv
    parameter_source.csv

outputs/
    parameter_library_primary_used.csv
"""

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# =========================================================
# Division by zero prevention
# =========================================================
EPS = 1e-12


# =========================================================
# Performance metrics
# =========================================================
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < EPS:
        return np.nan
    return float(1 - ss_res / ss_tot)


def aic(n: int, k: int, sse: float) -> float:
    sse = max(float(sse), EPS)
    return float(n * np.log(sse / n) + 2 * k)


def aicc(n: int, k: int, sse: float) -> float:
    a = aic(n, k, sse)
    denom = n - k - 1
    if denom <= 0:
        return np.inf
    return float(a + (2 * k * (k + 1)) / denom)


def ci95_from_seed_means(values: pd.Series) -> float:
    values = values.dropna()
    n = len(values)
    if n <= 1:
        return np.nan
    return float(1.96 * values.std(ddof=1) / np.sqrt(n))


# =========================================================
# Substrate-inhibition models
# =========================================================
def monod(S, mumax, Ks):
    return mumax * S / (Ks + S + EPS)


def haldane_andrews(S, mumax, Ks, Ki):
    return mumax * S / (Ks + S + (S**2 / (Ki + EPS)) + EPS)


def aiba_edwards(S, mumax, Ks, Ki):
    return mumax * (S / (Ks + S + EPS)) * np.exp(-S / (Ki + EPS))


def luong(S, mumax, Ks, Sm, n):
    inhib = np.clip(1 - S / (Sm + EPS), 0.0, None) ** n
    return mumax * (S / (Ks + S + EPS)) * inhib


def han_levenspiel(S, mumax, Ks, Sm, m, n):
    clipped = np.clip(1 - S / (Sm + EPS), 0.0, None)
    numerator_factor = clipped ** n
    denominator = S + Ks * (clipped ** m)
    return mumax * numerator_factor * (S / (denominator + EPS))



# =========================================================
# Model directory
# =========================================================
MODELS = {
    "Monod": {
        "func": monod,
        "p0": [0.4, 2.0],
        "bounds": ([1e-6, 1e-6], [10.0, 1e4]),
        "params": ["mumax", "Ks"],
    },
    "Haldane-Andrews": {
        "func": haldane_andrews,
        "p0": [0.4, 2.0, 30.0],
        "bounds": ([1e-6, 1e-6, 1e-6], [10.0, 1e4, 1e8]),
        "params": ["mumax", "Ks", "Ki"],
    },
    "Aiba-Edwards": {
        "func": aiba_edwards,
        "p0": [0.4, 2.0, 30.0],
        "bounds": ([1e-6, 1e-6, 1e-6], [10.0, 1e4, 1e8]),
        "params": ["mumax", "Ks", "Ki"],
    },
    "Luong": {
        "func": luong,
        "p0": [0.4, 2.0, 80.0, 2.0],
        "bounds": ([1e-6, 1e-6, 1e-6, 0.1], [10.0, 1e4, 1e6, 20.0]),
        "params": ["mumax", "Ks", "Sm", "n"],
    },
    "Han-Levenspiel": {
        "func": han_levenspiel,
        "p0": [0.4, 2.0, 80.0, 1.0, 2.0],
        "bounds": ([1e-6, 1e-6, 1e-6, 0.1, 0.1], [10.0, 1e4, 1e6, 20.0, 20.0]),
        "params": ["mumax", "Ks", "Sm", "m", "n"],
    },
}


# =========================================================
# Term decomposition for structural-failure analysis
# =========================================================
def term_decomposition(model_name: str, S: np.ndarray, p: Sequence[float]) -> Dict[str, np.ndarray]:

    if model_name == "Monod":
        mumax, Ks = p
        numerator = mumax * S
        denominator = Ks + S
        saturation_fraction = S / (denominator + EPS)
        total = numerator / (denominator + EPS)
        return {
            "numerator": numerator,
            "denominator": denominator,
            "saturation_fraction": saturation_fraction,
            "total": total,
        }

    if model_name == "Haldane-Andrews":
        mumax, Ks, Ki = p
        numerator = mumax * S
        base_denominator = Ks + S
        inhibition_denominator = S**2 / (Ki + EPS)
        denominator = base_denominator + inhibition_denominator
        total = numerator / (denominator + EPS)
        return {
            "numerator": numerator,
            "base_denominator": base_denominator,
            "inhibition_denominator": inhibition_denominator,
            "denominator": denominator,
            "total": total,
        }

    if model_name == "Aiba-Edwards":
        mumax, Ks, Ki = p
        numerator = mumax * S
        denominator = Ks + S
        saturation_fraction = S / (denominator + EPS)
        exponential_inhibition = np.exp(-S / (Ki + EPS))
        total = (numerator / (denominator + EPS)) * exponential_inhibition
        return {
            "numerator": numerator,
            "denominator": denominator,
            "saturation_fraction": saturation_fraction,
            "exponential_inhibition": exponential_inhibition,
            "total": total,
        }

    if model_name == "Luong":
        mumax, Ks, Sm, n = p
        numerator = mumax * S
        denominator = Ks + S
        saturation_fraction = S / (denominator + EPS)
        inhibition = np.clip(1 - S / (Sm + EPS), 0.0, None) ** n
        total = (numerator / (denominator + EPS)) * inhibition
        return {
            "numerator": numerator,
            "denominator": denominator,
            "saturation_fraction": saturation_fraction,
            "inhibition": inhibition,
            "total": total,
        }

    if model_name == "Han-Levenspiel":
        mumax, Ks, Sm, m, n = p
        clipped = np.clip(1 - S / (Sm + EPS), 0.0, None)
        inhibition_numerator = clipped ** n
        denominator_inhibition_factor = clipped ** m
        denominator = S + Ks * denominator_inhibition_factor
        prefactor = mumax * inhibition_numerator
        fraction = S / (denominator + EPS)
        total = prefactor * fraction
        return {
            "inhibition_numerator": inhibition_numerator,
            "denominator_inhibition_factor": denominator_inhibition_factor,
            "denominator": denominator,
            "prefactor": prefactor,
            "fraction": fraction,
            "total": total,
        }

    raise ValueError(f"Unknown model: {model_name}")


# =========================================================
# Data generation
# =========================================================
def make_substrate_grid(n_points: int = 80, s_min: float = 0.05, s_max: float = 80.0, logspace: bool = True) -> np.ndarray:
    if logspace:
        return np.logspace(np.log10(s_min), np.log10(s_max), n_points)
    return np.linspace(s_min, s_max, n_points)


def add_gaussian_noise(y_clean: np.ndarray, rel_noise: float, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    sigma = rel_noise * max(np.max(np.abs(y_clean)), EPS)
    y_noisy = y_clean + rng.normal(0.0, sigma, size=len(y_clean))
    return y_noisy, float(sigma)


# =========================================================
# Fitting
# =========================================================
def fit_one_model(model_name: str, S: np.ndarray, y: np.ndarray):
    spec = MODELS[model_name]
    try:
        popt, _ = curve_fit(
            spec["func"],
            S,
            y,
            p0=spec["p0"],
            bounds=spec["bounds"],
            maxfev=50000,
        )
        yhat = spec["func"](S, *popt)
        sse = float(np.sum((y - yhat) ** 2))

        out = {
            "success": True,
            "model": model_name,
            "sse": sse,
            "rmse": rmse(y, yhat),
            "r2": r2_manual(y, yhat),
            "aic": aic(len(y), len(popt), sse),
            "aicc": aicc(len(y), len(popt), sse),
            "params": dict(zip(spec["params"], popt)),
        }
        return out, popt, yhat

    except Exception as e:
        return {
            "success": False,
            "model": model_name,
            "sse": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "aic": np.nan,
            "aicc": np.nan,
            "params": {},
            "error": str(e),
        }, None, None


# =========================================================
# Main seed-ensemble forward-study loop
# =========================================================
def run_forward_noise_study(
    true_model_name: str,
    true_params: Sequence[float],
    noise_levels: Sequence[float],
    seed_values: Optional[Sequence[int]] = None,
    n_replicates_per_seed: int = 1,
    substrate_grid: Optional[np.ndarray] = None,
    save_pointwise: bool = True,
):
    if seed_values is None:
        seed_values = [42]
    seed_values = list(seed_values)

    S = make_substrate_grid() if substrate_grid is None else np.asarray(substrate_grid, dtype=float)

    true_func = MODELS[true_model_name]["func"]
    y_clean = true_func(S, *true_params)
    true_terms = term_decomposition(true_model_name, S, true_params)
    true_param_names = MODELS[true_model_name]["params"]
    true_param_map = dict(zip(true_param_names, true_params))

    rows: List[dict] = []
    point_rows: List[dict] = []

    for seed_index, seed in enumerate(seed_values):
        rng = np.random.default_rng(int(seed))

        for noise in noise_levels:
            for rep in range(n_replicates_per_seed):
                y_noisy, sigma = add_gaussian_noise(y_clean, noise, rng)
                global_replicate = seed_index * n_replicates_per_seed + rep

                for candidate in MODELS.keys():
                    fit_summary, popt, yhat = fit_one_model(candidate, S, y_noisy)

                    row = {
                        "true_model": true_model_name,
                        "candidate_model": candidate,
                        "noise_level": noise,
                        "seed_index": seed_index,
                        "random_seed": int(seed),
                        "replicate": rep,
                        "global_replicate": global_replicate,
                        "n_substrate_points": len(S),
                        "noise_sigma": sigma,
                        "success": fit_summary["success"],
                        "sse": fit_summary["sse"],
                        "rmse": fit_summary["rmse"],
                        "r2": fit_summary["r2"],
                        "aic": fit_summary["aic"],
                        "aicc": fit_summary["aicc"],
                    }

                    # Save fitted parameters.
                    for pname, pval in fit_summary["params"].items():
                        row[pname] = pval

                    if candidate == true_model_name and popt is not None:
                        for pname in true_param_names:
                            fitted_val = row.get(pname, np.nan)
                            true_val = true_param_map[pname]
                            row[f"{pname}_true"] = true_val
                            row[f"{pname}_abs_error"] = abs(fitted_val - true_val)
                            row[f"{pname}_rel_error"] = abs(fitted_val - true_val) / max(abs(true_val), EPS)

                    # Term-level diagnostics.
                    if popt is not None:
                        fit_terms = term_decomposition(candidate, S, popt)
                        common_keys = set(true_terms.keys()).intersection(fit_terms.keys())
                        for key in common_keys:
                            row[f"{key}_rmse"] = rmse(true_terms[key], fit_terms[key])

                        if save_pointwise:
                            for i, s_val in enumerate(S):
                                point_rows.append({
                                    "true_model": true_model_name,
                                    "candidate_model": candidate,
                                    "noise_level": noise,
                                    "seed_index": seed_index,
                                    "random_seed": int(seed),
                                    "replicate": rep,
                                    "global_replicate": global_replicate,
                                    "substrate_index": i,
                                    "S": float(s_val),
                                    "y_clean": float(y_clean[i]),
                                    "y_noisy": float(y_noisy[i]),
                                    "yhat": float(yhat[i]),
                                    "residual_to_noisy": float(y_noisy[i] - yhat[i]),
                                    "residual_to_clean": float(y_clean[i] - yhat[i]),
                                    "squared_error_noisy": float((y_noisy[i] - yhat[i]) ** 2),
                                    "squared_error_clean": float((y_clean[i] - yhat[i]) ** 2),
                                })

                    rows.append(row)

    results = pd.DataFrame(rows)
    pointwise = pd.DataFrame(point_rows) if save_pointwise else pd.DataFrame()
    substrate_df = pd.DataFrame({
        "substrate_index": np.arange(len(S)),
        "S": S,
        "y_clean": y_clean,
        "true_model": true_model_name,
    })

    return results, pointwise, substrate_df


# =========================================================
# Per-case runner and CSV export
# =========================================================
def run_case_and_save(
    true_model: str,
    true_params: Sequence[float],
    noise_levels: Sequence[float],
    seed_values: Sequence[int],
    n_replicates_per_seed: int = 1,
    substrate_grid: Optional[np.ndarray] = None,
    output_root: str = "outputs",
    save_pointwise: bool = True,
):
    case_dir = os.path.join(output_root, true_model)
    os.makedirs(case_dir, exist_ok=True)

    results, pointwise, substrate_df = run_forward_noise_study(
        true_model_name=true_model,
        true_params=true_params,
        noise_levels=noise_levels,
        seed_values=seed_values,
        n_replicates_per_seed=n_replicates_per_seed,
        substrate_grid=substrate_grid,
        save_pointwise=save_pointwise,
    )

    # Best model is selected independently for every seed/noise/replicate dataset.
    best = (
        results.sort_values(["noise_level", "seed_index", "replicate", "aicc"])
        .groupby(["noise_level", "seed_index", "random_seed", "replicate", "global_replicate"], as_index=False)
        .first()
    )

    # Overall summary across all seeds and replicates.
    summary = (
        results.groupby(["noise_level", "candidate_model"], as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            mean_r2=("r2", "mean"),
            mean_aicc=("aicc", "mean"),
            mean_total_rmse=("total_rmse", "mean"),
        )
        .sort_values(["noise_level", "mean_aicc"])
    )

    # Seed-level summary used for confidence intervals on curves.
    summary_by_seed_noise = (
        results.groupby(["seed_index", "random_seed", "noise_level", "candidate_model"], as_index=False)
        .agg(
            seed_mean_rmse=("rmse", "mean"),
            seed_mean_r2=("r2", "mean"),
            seed_mean_aicc=("aicc", "mean"),
            seed_mean_total_rmse=("total_rmse", "mean"),
        )
    )

    summary_with_ci = (
        summary_by_seed_noise.groupby(["noise_level", "candidate_model"], as_index=False)
        .agg(
            n_seeds=("seed_index", "nunique"),
            mean_rmse=("seed_mean_rmse", "mean"),
            ci95_rmse=("seed_mean_rmse", ci95_from_seed_means),
            mean_r2=("seed_mean_r2", "mean"),
            ci95_r2=("seed_mean_r2", ci95_from_seed_means),
            mean_aicc=("seed_mean_aicc", "mean"),
            ci95_aicc=("seed_mean_aicc", ci95_from_seed_means),
            mean_total_rmse=("seed_mean_total_rmse", "mean"),
            ci95_total_rmse=("seed_mean_total_rmse", ci95_from_seed_means),
        )
        .sort_values(["noise_level", "mean_aicc"])
    )

    metadata = pd.DataFrame([
        {
            "true_model": true_model,
            "true_params": repr(list(true_params)),
            "noise_levels": repr(list(noise_levels)),
            "seed_values": repr(list(seed_values)),
            "n_seed_values": len(seed_values),
            "n_replicates_per_seed": n_replicates_per_seed,
            "total_noisy_datasets_per_noise": len(seed_values) * n_replicates_per_seed,
            "n_substrate_points": len(substrate_df),
            "substrate_min": substrate_df["S"].min(),
            "substrate_max": substrate_df["S"].max(),
            "substrate_spacing": "logspace",
            "pointwise_predictions_saved": bool(save_pointwise),
        }
    ])

    results.to_csv(os.path.join(case_dir, "all_fits.csv"), index=False)
    best.to_csv(os.path.join(case_dir, "best_by_replicate.csv"), index=False)
    summary.to_csv(os.path.join(case_dir, "summary_by_noise.csv"), index=False)
    summary_by_seed_noise.to_csv(os.path.join(case_dir, "summary_by_seed_noise.csv"), index=False)
    summary_with_ci.to_csv(os.path.join(case_dir, "summary_with_ci.csv"), index=False)
    substrate_df.to_csv(os.path.join(case_dir, "substrate_grid.csv"), index=False)
    metadata.to_csv(os.path.join(case_dir, "study_metadata.csv"), index=False)

    if save_pointwise:
        pointwise.to_csv(os.path.join(case_dir, "pointwise_predictions.csv"), index=False)

    print(f"\nFinished case: {true_model}")
    print(f"Saved to: {case_dir}")
    print(f"Seeds used: {list(seed_values)}")
    print(f"Noisy datasets per noise level: {len(seed_values) * n_replicates_per_seed}")

    print("\nBest model counts by noise level:")
    print(best.groupby(["noise_level", "candidate_model"]).size().reset_index(name="count"))

    return results, best, summary, summary_with_ci, substrate_df, pointwise


# =========================================================
# Batch run for active truth models using literature parameter library
# =========================================================
def load_literature_true_cases(parameter_library_path: str = "literature_parameter_sets.csv"):

    library = pd.read_csv(parameter_library_path)
    primary = library[library["parameter_status"].astype(str).str.lower() == "primary"].copy()
    primary = primary[primary["model"].astype(str).isin(MODELS.keys())].copy()

    missing_models = [name for name in MODELS.keys() if name not in primary["model"].tolist()]
    if missing_models:
        raise ValueError(
            "The literature parameter library is missing primary rows for: "
            + ", ".join(missing_models)
        )

    cases = {}
    for _, row in primary.iterrows():
        model_name = row["model"]
        param_names = MODELS[model_name]["params"]
        params = []
        for pname in param_names:
            if pname not in row or pd.isna(row[pname]) or row[pname] == "":
                raise ValueError(
                    f"Missing parameter '{pname}' for model '{model_name}' "
                    f"in {parameter_library_path}"
                )
            params.append(float(row[pname]))

        cases[model_name] = {
            "params": params,
            "s_min": float(row["S_min"]),
            "s_max": float(row["S_max"]),
            "parameter_set_id": row.get("parameter_set_id", ""),
            "source_short": row.get("source_short", ""),
            "source_url": row.get("source_url", ""),
            "S_units": row.get("S_units", ""),
        }

    return cases, primary


if __name__ == "__main__":
    NOISE_LEVELS = [0.00, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    # Thirty documented seeds. Change to range(42, 62) for 20 seeds if desired.
    SEED_VALUES = list(range(42, 72))

    N_REPLICATES_PER_SEED = 1

    PARAMETER_LIBRARY_FILE = "literature_parameter_sets.csv"
    TRUE_CASES, PARAMETER_LIBRARY_PRIMARY = load_literature_true_cases(PARAMETER_LIBRARY_FILE)

    os.makedirs("outputs", exist_ok=True)
    PARAMETER_LIBRARY_PRIMARY.to_csv(
        os.path.join("outputs", "parameter_library_primary_used.csv"),
        index=False,
    )

    for true_model, case in TRUE_CASES.items():
        true_params = case["params"]
        substrate_grid = make_substrate_grid(
            s_min=case.get("s_min", 0.05),
            s_max=case.get("s_max", 80.0),
        )

        run_case_and_save(
            true_model=true_model,
            true_params=true_params,
            noise_levels=NOISE_LEVELS,
            seed_values=SEED_VALUES,
            n_replicates_per_seed=N_REPLICATES_PER_SEED,
            substrate_grid=substrate_grid,
            output_root="outputs",
            save_pointwise=True,
        )

        # Save a copy of the literature source row inside each model output folder.
        source_row = PARAMETER_LIBRARY_PRIMARY[PARAMETER_LIBRARY_PRIMARY["model"] == true_model]
        source_row.to_csv(os.path.join("outputs", true_model, "parameter_source.csv"), index=False)

    print("\nAll seed-ensemble literature-parameter truth cases completed.")
