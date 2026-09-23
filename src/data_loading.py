import pickle

import h5py
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components


def require_data(directory, filenames):
    missing = [directory / name for name in filenames if not (directory / name).is_file()]
    if missing:
        raise SystemExit("Data not found. Please add these files:\n" + "\n".join(map(str, missing)))


def split_boundaries(length):
    return [int(length * fraction) for fraction in (0, 0.6, 0.7, 0.8, 0.9, 1)]


def load_traffic(directory, speed_file, graph_file):
    with (directory / graph_file).open("rb") as stream:
        ids, _, adjacency = pickle.load(stream, encoding="latin1")
    ids = [str(value) for value in ids]
    with h5py.File(directory / speed_file, "r") as stream:
        group = next(group for group in stream.values()
                     if isinstance(group, h5py.Group) and "block0_values" in group)
        columns = [value.decode() if isinstance(value, bytes) else str(value)
                   for value in group["axis0"][...]]
        frame = pd.DataFrame(group["block0_values"][...].astype(np.float32),
                             index=pd.to_datetime(group["axis1"][...], unit="ns"),
                             columns=columns).sort_index().loc[:, ids]
    speed = frame.to_numpy(dtype=np.float32, copy=True)
    observed = np.isfinite(speed) & (speed > 0)
    boundaries = split_boundaries(len(frame))
    masked = frame.mask(~observed)
    training = masked.iloc[:boundaries[1]]
    medians = training.median().fillna(float(np.nanmedian(training.to_numpy())))
    filled = masked.ffill().fillna(medians).to_numpy(dtype=np.float32)
    training_values = speed[:boundaries[1]][observed[:boundaries[1]]]
    mean, deviation = float(training_values.mean()), float(training_values.std())
    speed[~observed] = np.nan
    directed = np.asarray(adjacency, dtype=np.float32)
    graph = np.maximum(directed.astype(float), directed.T.astype(float))
    np.fill_diagonal(graph, 0)
    return {"ids": ids, "dates": frame.index.to_numpy(dtype="datetime64[ns]"),
            "speed": speed, "observed": observed, "graph": graph,
            "forecast_graph": directed, "inputs": (filled - mean) / deviation,
            "mean": mean, "deviation": deviation, "boundaries": boundaries}


def load_locations(path, ids):
    frame = pd.read_csv(path, dtype={"sensor_id": str})
    frame.columns = frame.columns.str.strip().str.lower()
    if "sensor_id" not in frame.columns:
        frame = pd.read_csv(path, header=None, names=["sensor_id", "latitude", "longitude"],
                            dtype={"sensor_id": str})
    frame["sensor_id"] = frame["sensor_id"].astype(str).str.strip()
    return frame.set_index("sensor_id").loc[ids, ["latitude", "longitude"]]


def load_hospital(directory):
    frame = pd.read_csv(directory / "HHShosp.csv", index_col=0, parse_dates=True).sort_index()
    adjacency = pd.read_csv(directory / "HHShosp_adj.csv", index_col=0)
    graph = adjacency.loc[frame.columns, frame.columns].to_numpy(dtype=float)
    np.fill_diagonal(graph, 0)
    _, labels = connected_components(graph, directed=False)
    largest = np.argmax(np.bincount(labels))
    ids = sorted(frame.columns[labels == largest])
    frame = frame.loc[:, ids].reindex(pd.date_range(frame.index.min(), frame.index.max()))
    counts = frame.to_numpy(dtype=np.float32)
    graph = adjacency.loc[ids, ids].to_numpy(dtype=float)
    np.fill_diagonal(graph, 0)
    boundaries = split_boundaries(len(frame))
    deviation = counts[:boundaries[1]].std(axis=0, dtype=np.float64)
    factors = np.where(deviation > 1e-6, deviation, 1).astype(np.float32).astype(float)
    return {"ids": ids, "dates": frame.index.to_numpy(dtype="datetime64[ns]"),
            "counts": counts, "graph": graph, "factors": factors,
            "boundaries": boundaries}


def hospital_forecasts(data, split, horizon):
    start, end = data["boundaries"][split:split + 2]
    issues = np.arange(start - 1, end - horizon)
    targets = issues + horizon
    predictions = data["counts"][issues].astype(float)
    observations = data["counts"][targets].astype(float)
    return {"predictions": predictions, "observations": observations,
            "differences": np.abs(predictions - observations) / data["factors"],
            "issues": issues, "targets": targets}
