import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from tqdm.auto import tqdm, trange


_CONTEXT = None


def scale_quantiles(values):
    counts = np.isfinite(values).sum(axis=0)
    scales = np.full(values.shape[1], np.nan)
    scales[counts > 0] = np.nanquantile(values[:, counts > 0], 0.9, axis=0, method="higher")
    return counts, scales


def variance_coefficients(history, window, repetitions, seed):
    length = len(history)
    window = length if window is None else min(window, length)
    sampled = min(window, length // 2)
    generator = np.random.default_rng(seed)
    logs = np.empty((repetitions, history.shape[1]))
    counts = np.empty_like(logs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for repetition in trange(repetitions, desc=f"Scale variance ({window} rows)",
                                 leave=False, dynamic_ncols=True):
            start = generator.integers(0, length - sampled + 1)
            values = history[start:start + sampled]
            # Preserve the original resampling sequence without saving unused diagnostics.
            generator.choice(length, size=sampled, replace=False)
            counts[repetition] = np.isfinite(values).sum(axis=0)
            logs[repetition] = np.log(np.maximum(
                np.nanquantile(values, 0.9, axis=0, method="higher"), 1e-6))
        variance = np.nanvar(logs, axis=0, ddof=1)
    variance = np.where(np.isfinite(variance), np.maximum(variance, 1e-8), np.inf)
    variance *= sampled / window
    reference_count = (np.isfinite(history).sum(axis=0) if window == length else
                       np.rint(np.median(counts, axis=0) * window / sampled))
    coefficients = np.full(history.shape[1], np.inf)
    usable = (reference_count > 0) & np.isfinite(variance)
    coefficients[usable] = reference_count[usable] * variance[usable]
    return coefficients


def build_context(ids, graph, old, recent, calibration, histories, repetitions, seed,
                  factors, alpha, clip_zero, evaluation=None):
    _, labels = connected_components(csr_matrix(graph > 0), directed=False)
    references = {}
    for position, (history, window) in enumerate(list(histories.items())[1:]):
        references[history] = variance_coefficients(old, window, repetitions, seed + position)
    estimation = np.concatenate((old, recent))
    counts, scales = scale_quantiles(estimation)
    fallback = []
    for target in range(len(ids)):
        values = np.delete(estimation, target, axis=1)
        values = values[np.isfinite(values)]
        fallback.append(np.log(max(float(np.quantile(values, 0.9, method="higher")), 1e-6)))
    return {"ids": ids, "graph": graph, "labels": labels, "histories": histories,
            "recent": recent, "calibration": calibration, "references": references,
            "counts": counts, "logs": np.log(np.maximum(np.where(np.isfinite(scales), scales, 1), 1e-6)),
            "weights": counts / references["full"], "fallback": np.array(fallback),
            "factors": factors, "alpha": alpha, "clip_zero": clip_zero,
            "evaluation": evaluation}


def strengths_for_ratios(weights, graph, labels, ratios):
    eigenvalues = []
    degrees = graph.sum(axis=1)
    for component in np.unique(labels):
        indices = np.flatnonzero(labels == component)
        local = weights[indices]
        observed, missing = np.flatnonzero(local > 0), np.flatnonzero(local <= 0)
        if not len(observed):
            continue
        laplacian = np.diag(degrees[indices]) - graph[np.ix_(indices, indices)]
        reduced = laplacian[np.ix_(observed, observed)]
        if len(missing):
            coupling = laplacian[np.ix_(observed, missing)]
            reduced = reduced - coupling @ np.linalg.solve(laplacian[np.ix_(missing, missing)], coupling.T)
        roots = np.sqrt(local[observed])
        normalized = reduced / roots[:, None] / roots[None, :]
        spectrum = np.maximum(np.linalg.eigvalsh((normalized + normalized.T) / 2), 0)
        spectrum[0] = 0
        eigenvalues.extend(spectrum)
    spectrum = np.asarray(eigenvalues)
    minimum = float(np.count_nonzero(spectrum == 0))
    strengths = {}
    for ratio in ratios:
        if ratio == 1 or minimum == len(spectrum):
            strengths[ratio] = 0.0
            continue
        target = max(ratio * len(spectrum), minimum + 1e-4 * max(1, len(spectrum) - minimum))
        low, high = 0.0, 1.0
        while np.sum(1 / (1 + high * spectrum)) > target:
            high *= 10
        for _ in range(40):
            middle = high / 2 if low == 0 else float(np.sqrt(low * high))
            if np.sum(1 / (1 + middle * spectrum)) > target:
                low = middle
            else:
                high = middle
        strengths[ratio] = high
    return strengths


def smooth_scales(logs, weights, graph, labels, strength, fallback):
    if strength == 0:
        return np.exp(np.where(weights > 0, logs, fallback))
    laplacian = diags(graph.sum(axis=1)) - csr_matrix(graph)
    smoothed = np.full_like(logs, fallback)
    for component in np.unique(labels):
        indices = np.flatnonzero(labels == component)
        local = weights[indices]
        if np.any(local > 0):
            system = diags(local) + strength * laplacian[indices][:, indices]
            smoothed[indices] = spsolve(system.tocsr(), local * logs[indices])
    return np.exp(smoothed)


def conformal_quantile(values, alpha):
    values = values[np.isfinite(values)]
    rank = int(np.ceil((len(values) + 1) * (1 - alpha)))
    return float(np.partition(values, rank - 1)[rank - 1]) if rank <= len(values) else float("inf")


def metrics(evaluation, target, radius, factor, alpha, clip_zero):
    differences = evaluation["differences"][:, target]
    usable = np.isfinite(differences)
    if clip_zero:
        predicted = evaluation["predictions"][usable, target]
        observed = evaluation["observations"][usable, target]
        lower, upper = np.maximum(0, predicted - radius), predicted + radius
        widths = upper - lower
        scores = widths + (2 / alpha) * (np.maximum(lower - observed, 0) + np.maximum(observed - upper, 0))
        coverage = np.mean((observed >= lower) & (observed <= upper))
        width = float(widths.mean())
    else:
        values = differences[usable]
        scores = 2 * radius + (2 / alpha) * np.maximum(values - radius, 0)
        coverage, width = np.mean(values <= radius), 2 * radius
    return {"observations": int(usable.sum()), "coverage": float(coverage),
            "width": width, "winkler_score": float(scores.mean()),
            "scaled_winkler": float(scores.mean() / factor)}


def initialize_worker(context):
    global _CONTEXT
    _CONTEXT = context


def fit_case(task):
    target, history, ratios = task
    context = _CONTEXT
    donors = np.arange(len(context["ids"])) != target
    logs, weights = context["logs"].copy(), context["weights"].copy()
    fallback = context["fallback"][target]
    logs[weights <= 0] = fallback
    count = int(context["counts"][target])
    if history != "full":
        window = context["histories"][history]
        values = context["recent"][-window:, target] if window else np.array([])
        values = values[np.isfinite(values)]
        count = len(values)
        logs[target], weights[target] = fallback, 0.0
        if count:
            logs[target] = np.log(max(float(np.quantile(values, 0.9, method="higher")), 1e-6))
            coefficients = context["references"][history]
            usable = donors & np.isfinite(coefficients) & (coefficients > 0)
            weights[target] = count / float(np.median(coefficients[usable]))
    requested = tuple(dict.fromkeys((1.0, *ratios)))
    strengths = strengths_for_ratios(weights, context["graph"], context["labels"], requested)
    calibration = context["calibration"][:, donors]
    factor = float(context["factors"][target])
    rows = []

    def add(method, ratio, strength, scale, multiplier):
        radius = float(scale * multiplier * factor)
        row = {"location": context["ids"][target], "history": history, "method": method,
               "ratio": ratio, "lambda": strength, "available_count": count,
               "normalization": factor, "scale": scale, "multiplier": multiplier, "radius": radius}
        if context["evaluation"] is not None:
            row.update(metrics(context["evaluation"], target, radius, factor,
                               context["alpha"], context["clip_zero"]))
        rows.append(row)

    add("Global", np.nan, 0.0, 1.0, conformal_quantile(calibration, context["alpha"]))
    for ratio in requested:
        strength = strengths[ratio]
        scales = smooth_scales(logs, weights, context["graph"], context["labels"], strength, fallback)
        multiplier = conformal_quantile(calibration / scales[None, donors], context["alpha"])
        if ratio == 1:
            add("Local", ratio, strength, float(scales[target]), multiplier)
        if ratio in ratios:
            add("Graph Smoothing", ratio, strength, float(scales[target]), multiplier)
    return rows


def fit_cases(context, ratios, workers, label):
    tasks = [(target, history, tuple(ratios[history])) for history in context["histories"]
             for target in range(len(context["ids"]))]
    rows = []
    if workers == 1:
        initialize_worker(context)
        for result in tqdm(map(fit_case, tasks), total=len(tasks), desc=label,
                           unit="location", dynamic_ncols=True):
            rows.extend(result)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=initialize_worker,
                                 initargs=(context,)) as pool:
            for result in tqdm(pool.map(fit_case, tasks), total=len(tasks), desc=label,
                               unit="location", dynamic_ncols=True):
                rows.extend(result)
    return pd.DataFrame(rows)


def select_ratios(rows, histories, tolerance):
    grid = rows[rows.method == "Graph Smoothing"].groupby(["history", "ratio"])["scaled_winkler"].mean()
    choices = {}
    for history in histories:
        scores = grid.loc[history]
        choices[history] = float(scores[scores <= scores.min() * (1 + tolerance)].index.max())
    return choices


def evaluate_cases(fitted, evaluation, ids, factors, alpha, clip_zero):
    positions = {location: index for index, location in enumerate(ids)}
    rows = fitted.to_dict("records")
    for row in rows:
        target = positions[row["location"]]
        row.update(metrics(evaluation, target, float(row["radius"]), float(factors[target]), alpha, clip_zero))
    return pd.DataFrame(rows)


def average_results(rows):
    return rows.groupby(["dataset", "horizon", "history", "method"], sort=False)[
        ["coverage", "width", "winkler_score", "scaled_winkler"]].mean().reset_index()
