import os
from pathlib import Path


FIGURES_ONLY = True
WORKERS = 16
SEED = 2026
BOOTSTRAPS = 200
ALPHA = 0.10
SCORE_TOLERANCE = 0.01
HORIZON = 7
HISTORIES = {"0": 0, "7": 7, "14": 14, "full": None}
RATIOS = (1.0, 0.85, 0.70, 0.58, 0.47, 0.38, 0.31, 0.25,
          0.20, 0.16, 0.12, 0.09, 0.06, 0.03, 0.0)
OLDER_TUNING_DAYS = 32
SELECTION_DAYS = 21
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULT = ROOT / "result"

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from data_loading import hospital_forecasts, load_hospital, require_data
from paper_outputs import hospital_outputs, regenerate_figures
from prediction_intervals import (average_results, build_context, evaluate_cases,
                                  fit_cases, select_ratios)


def main():
    RESULT.mkdir(parents=True, exist_ok=True)
    if FIGURES_ONLY:
        regenerate_figures("hospital", RESULT)
        return
    require_data(DATA, ["HHShosp.csv", "HHShosp_adj.csv"])
    data = load_hospital(DATA)
    development = hospital_forecasts(data, 2, HORIZON)
    recent_days = max(window for window in HISTORIES.values() if window is not None)
    selection_start = len(development["targets"]) - SELECTION_DAYS
    cutoff = development["issues"][selection_start]
    available = np.flatnonzero((np.arange(len(development["targets"])) >= OLDER_TUNING_DAYS)
                               & (development["targets"] <= cutoff))
    values = development["differences"]
    evaluation = {key: development[key][selection_start:]
                  for key in ("predictions", "observations", "differences")}
    context = build_context(data["ids"], data["graph"], values[:OLDER_TUNING_DAYS],
                            values[available[-recent_days:]], values[available[:-recent_days]],
                            HISTORIES, BOOTSTRAPS, SEED, data["factors"], ALPHA, True, evaluation)
    grid = fit_cases(context, {history: RATIOS for history in HISTORIES}, WORKERS, "Hospital: tune")
    choices = select_ratios(grid, HISTORIES, SCORE_TOLERANCE)
    print(f"Hospital: selected ratios {choices}", flush=True)

    calibration = hospital_forecasts(data, 3, HORIZON)
    matured = calibration["targets"] <= data["boundaries"][4] - 1
    available = calibration["differences"][matured]
    context = build_context(data["ids"], data["graph"], values,
                            available[-recent_days:], available[:-recent_days], HISTORIES,
                            BOOTSTRAPS, SEED + 1000, data["factors"], ALPHA, True)
    fitted = fit_cases(context, {history: (choices[history],) for history in HISTORIES},
                       WORKERS, "Hospital: calibrate")
    test = hospital_forecasts(data, 4, HORIZON)
    results = evaluate_cases(fitted, test, data["ids"], data["factors"], ALPHA, True)
    results["dataset"], results["horizon"] = "Hospital admissions", HORIZON
    hospital_outputs(data, results, average_results(results), HISTORIES, RESULT)
    print(f"Hospital paper outputs saved to {RESULT}", flush=True)


if __name__ == "__main__":
    main()
