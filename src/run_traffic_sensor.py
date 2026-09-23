import os
from pathlib import Path


FIGURES_ONLY = False
WORKERS = 16
SEED = 2026
BOOTSTRAPS = 200
ALPHA = 0.10
SCORE_TOLERANCE = 0.01
HORIZONS = (15, 30, 60)
HISTORIES = {"0": 0, "12": 12, "24": 24, "48": 48, "full": None}
RATIOS = (1.0, 0.85, 0.70, 0.58, 0.47, 0.38, 0.31, 0.25,
          0.20, 0.16, 0.12, 0.09, 0.06, 0.03, 0.0)
TRAINING = {"epochs": 100, "batch_size": 32, "learning_rate": 1e-3,
            "weight_decay": 1e-5, "patience": 10, "seed": SEED,
            "loader_workers": 0, "cpu_threads": 4}
PREDICTION_BATCH_SIZE = 64
DATASETS = (("Los Angeles", "metr-la.h5", "adj_mx.pkl",
             "graph_sensor_locations.csv", "traffic_network"),
            ("Bay Area", "pems-bay.h5", "adj_mx_bay.pkl",
             "graph_sensor_locations_bay.csv", "traffic_network_bay"))
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULT = ROOT / "result"

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd

from data_loading import load_locations, load_traffic, require_data
from paper_outputs import (regenerate_figures, sensor_improvements, traffic_example, traffic_network,
                           traffic_scores, traffic_tables)
from prediction_intervals import (average_results, build_context, evaluate_cases,
                                  fit_cases, select_ratios)


def fit_intervals(data, development, calibration, name):
    old_end = int(0.60 * len(development["differences"]))
    calibration_start = old_end + 12
    selection_start = int(0.80 * len(development["differences"])) + 12
    tuning_cutoff = development["issues"][selection_start]
    final_cutoff = data["boundaries"][4] - 1
    factors = np.ones(len(data["ids"]))
    final_rows = []
    recent_rows = max(window for window in HISTORIES.values() if window is not None)
    for position, horizon in enumerate(HORIZONS):
        values = development["differences"][:, position]
        available = values[calibration_start:selection_start]
        matured = development["targets"][calibration_start:selection_start, position] <= tuning_cutoff
        available = available[matured]
        context = build_context(data["ids"], data["graph"], values[:old_end],
                                available[-recent_rows:], available[:-recent_rows], HISTORIES,
                                BOOTSTRAPS, SEED + 100 * horizon, factors, ALPHA, False,
                                {"differences": values[selection_start:]})
        grid = fit_cases(context, {history: RATIOS for history in HISTORIES}, WORKERS,
                         f"{name}: tune {horizon} min")
        choices = select_ratios(grid, HISTORIES, SCORE_TOLERANCE)
        print(f"{name}, {horizon} min: selected ratios {choices}", flush=True)
        matured = calibration["targets"][:, position] <= final_cutoff
        available = calibration["differences"][matured, position]
        context = build_context(data["ids"], data["graph"], values,
                                available[-recent_rows:], available[:-recent_rows], HISTORIES,
                                BOOTSTRAPS, SEED + 100 * horizon, factors, ALPHA, False)
        fitted = fit_cases(context, {history: (choices[history],) for history in HISTORIES}, WORKERS,
                           f"{name}: calibrate {horizon} min")
        fitted["dataset"], fitted["horizon"] = name, horizon
        final_rows.append(fitted)
    return pd.concat(final_rows, ignore_index=True)


def main():
    RESULT.mkdir(parents=True, exist_ok=True)
    if FIGURES_ONLY:
        regenerate_figures("traffic", RESULT)
        return
    from stgcn import predict, train

    require_data(DATA, [filename for _, speed, graph, locations, _ in DATASETS
                       for filename in (speed, graph, locations)])
    all_rows, summaries = [], []
    for name, speed_file, graph_file, locations_file, network_file in DATASETS:
        data = load_traffic(DATA, speed_file, graph_file)
        traffic_network(data, load_locations(DATA / locations_file, data["ids"]), RESULT, network_file)
        summaries.append({"Dataset": name, "Sensors": len(data["ids"]),
                          "Time points": len(data["speed"]),
                          "Missing (%)": round(100 * (1 - data["observed"].mean()), 3),
                          "Edges": int(np.count_nonzero(np.triu(data["graph"], 1)))})
        fitted_model = train(data, TRAINING, name)
        development = predict(data, fitted_model, 2, PREDICTION_BATCH_SIZE, HORIZONS)
        calibration = predict(data, fitted_model, 3, PREDICTION_BATCH_SIZE, HORIZONS)
        fitted_intervals = fit_intervals(data, development, calibration, name)
        test = predict(data, fitted_model, 4, PREDICTION_BATCH_SIZE, HORIZONS)
        dataset_rows = []
        for position, horizon in enumerate(HORIZONS):
            rows = evaluate_cases(fitted_intervals[fitted_intervals.horizon == horizon],
                                  {"differences": test["differences"][:, position]}, data["ids"],
                                  np.ones(len(data["ids"])), ALPHA, False)
            dataset_rows.append(rows)
        results = pd.concat(dataset_rows, ignore_index=True)
        all_rows.append(results)
        if name == "Los Angeles":
            traffic_example(results, test, data, HORIZONS, RESULT)
        del fitted_model, development, calibration, test, data
    results = pd.concat(all_rows, ignore_index=True)
    aggregate = average_results(results)
    results.to_csv(RESULT / "traffic_location_results.csv", index=False)
    aggregate.to_csv(RESULT / "traffic_results.csv", index=False)
    traffic_tables(summaries, aggregate, RESULT)
    traffic_scores(aggregate, HISTORIES, HORIZONS, RESULT)
    sensor_improvements(results, HISTORIES, HORIZONS, RESULT)
    print(f"Traffic paper outputs saved to {RESULT}", flush=True)


if __name__ == "__main__":
    main()
