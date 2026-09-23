import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


METHODS = ("Global", "Local", "Graph Smoothing")
COLORS = {"Global": "#6B7280", "Local": "#2878B5", "Graph Smoothing": "#E58B2B"}
MARKERS = {"Global": "^", "Local": "o", "Graph Smoothing": "s"}
HORIZON_COLORS = {15: "#0072B2", 30: "#E69F00", 60: "#009E73"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def save_table(frame, path):
    frame.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".txt").write_text(frame.to_string(index=False) + "\n", encoding="utf-8")


def save_figure(figure, path):
    figure.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def traffic_network(data, locations, output, filename):
    points = locations[["longitude", "latitude"]].to_numpy()
    graph = data["graph"]
    rows, columns = np.nonzero(np.triu(graph > 0, k=1))
    missing = 1 - data["observed"].mean(axis=0)
    nodes = pd.DataFrame({"sensor": data["ids"], "longitude": points[:, 0], "latitude": points[:, 1],
                          "missing_fraction": missing})
    edges = pd.DataFrame({"source": np.array(data["ids"])[rows], "target": np.array(data["ids"])[columns],
                          "weight": graph[rows, columns]})
    nodes.to_csv(output / f"{filename}_nodes.csv", index=False)
    edges.to_csv(output / f"{filename}_edges.csv", index=False)
    plot_network(nodes, edges, output, filename)


def plot_network(nodes, edges, output, filename):
    locations = nodes.set_index("sensor")[["longitude", "latitude"]]
    segments = np.stack((locations.loc[edges.source].to_numpy(),
                         locations.loc[edges.target].to_numpy()), axis=1)
    figure, axis = plt.subplots(figsize=(3.35, 2.65), layout="constrained")
    axis.add_collection(LineCollection(segments, colors="#858B94", linewidths=0.35, alpha=0.22))
    dots = axis.scatter(nodes.longitude, nodes.latitude, c=nodes.missing_fraction, cmap="magma", s=8, zorder=3)
    figure.colorbar(dots, ax=axis, label="Missing fraction", fraction=0.045, pad=0.03)
    axis.set(xlabel="Longitude", ylabel="Latitude")
    axis.grid(alpha=0.12)
    save_figure(figure, output / f"{filename}.pdf")


def traffic_scores(aggregate, histories, horizons, output):
    figure, axes = plt.subplots(1, len(horizons), figsize=(7, 2.35), squeeze=False)
    labels = ["Full" if history == "full" else history for history in histories]
    for axis, horizon in zip(axes[0], horizons):
        for method in METHODS:
            values = aggregate[(aggregate.dataset == "Los Angeles") & (aggregate.horizon == horizon)
                               & (aggregate.method == method)].set_index("history").loc[list(histories)]
            axis.plot(range(len(histories)), values.winkler_score, color=COLORS[method],
                      marker=MARKERS[method], markersize=3.5, linewidth=1.2, label=method)
        axis.set_xticks(range(len(histories)), labels)
        axis.set(title=f"{horizon} minutes ahead", xlabel="Available five-minute records")
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Winkler score (mph)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1))
    figure.text(0.98, 0.98, "Lower is better", ha="right", va="top", fontsize=7)
    figure.tight_layout(rect=(0, 0, 1, 0.87))
    save_figure(figure, output / "traffic_winkler_by_history.pdf")


def sensor_improvements(results, histories, horizons, output, save_csv=True):
    paired = results.pivot(index=["dataset", "horizon", "history", "location"],
                           columns="method", values="winkler_score").reset_index()
    paired["improvement_percent"] = 100 * (1 - paired["Graph Smoothing"] / paired["Global"])
    paired.loc[paired.Global <= 0, "improvement_percent"] = np.nan
    if save_csv:
        paired.to_csv(output / "traffic_sensor_improvements.csv", index=False)
    histories = [history for history in histories if history != "full"]
    figure, axis = plt.subplots(figsize=(3.35, 2.65))
    for offset, horizon in zip(np.linspace(-0.24, 0.24, len(horizons)), horizons):
        color = HORIZON_COLORS[horizon]
        for position, history in enumerate(histories):
            values = paired.loc[(paired.dataset == "Los Angeles") & (paired.horizon == horizon)
                                & (paired.history == history), "improvement_percent"].dropna()
            axis.boxplot([values.to_numpy()], positions=[position + offset], widths=0.21,
                         patch_artist=True, manage_ticks=False,
                         boxprops={"facecolor": color, "alpha": 0.55},
                         medianprops={"color": "black", "linewidth": 0.8},
                         flierprops={"marker": ".", "markersize": 2, "alpha": 0.4,
                                     "markeredgecolor": color})
            for outside, boundary, marker, shift in (
                (values[values < -30], -29, "v", 7), (values[values > 60], 59, "^", -14)
            ):
                if len(outside):
                    label = f"{outside.iloc[0]:.0f}%" if len(outside) == 1 else f"{len(outside)} outside"
                    axis.scatter(position + offset, boundary, marker=marker, color=color, s=12, zorder=5)
                    axis.annotate(label, (position + offset, boundary), xytext=(0, shift),
                                  textcoords="offset points", ha="center", fontsize=5)
    axis.axhline(0, color="0.4", linestyle="--", linewidth=0.7)
    axis.set_xticks(range(len(histories)), histories)
    axis.set(xlim=(-0.6, len(histories) - 0.4), ylim=(-30, 60),
             xlabel="Available five-minute records", ylabel="Winkler reduction vs Global (%)")
    axis.grid(axis="y", alpha=0.2)
    handles = [Patch(facecolor=HORIZON_COLORS[horizon], alpha=0.55, label=f"{horizon} min") for horizon in horizons]
    figure.legend(handles=handles, loc="upper center", ncol=len(horizons), frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    save_figure(figure, output / "traffic_sensor_improvement_vs_global.pdf")


def traffic_example(results, test, data, horizons, output):
    horizon, history = 30, "12"
    position = horizons.index(horizon)
    candidates = results[(results.horizon == horizon) & (results.history == history)].pivot(
        index="location", columns="method", values="winkler_score").dropna()
    candidates["gain"] = np.minimum(1 - candidates["Graph Smoothing"] / candidates.Global,
                                    1 - candidates["Graph Smoothing"] / candidates.Local)
    positive = candidates[candidates.gain > 0]
    if not positive.empty:
        candidates = positive
    candidates = candidates.reset_index().sort_values(["gain", "location"], ascending=[False, True])
    times = data["dates"][test["targets"][:, position]]
    best_location, longest = None, np.array([], dtype=int)
    # This illustration is selected using test scores, as disclosed in the paper.
    for location in candidates.location:
        index = data["ids"].index(location)
        usable = np.flatnonzero(np.isfinite(test["differences"][:, position, index]))
        breaks = np.flatnonzero((np.diff(usable) != 1) | (np.diff(times[usable]) != np.timedelta64(5, "m"))) + 1
        for run in np.split(usable, breaks):
            if len(run) > len(longest):
                best_location, longest = location, run
            if len(run) >= 288:
                best_location, longest = location, run[:288]
                break
        if len(longest) >= 288:
            break
    if best_location is None:
        return
    selected, location = longest, best_location
    index = data["ids"].index(location)
    dates = pd.to_datetime(times[selected])
    predicted = test["predictions"][selected, position, index]
    observed = test["observations"][selected, position, index]
    fitted = results[(results.location == location) & (results.horizon == horizon)
                     & (results.history == history)].set_index("method")
    radii = {method: float(fitted.loc[method, "radius"]) for method in METHODS}
    saved = pd.DataFrame({"sensor": location, "horizon_minutes": horizon, "history": history,
                          "timestamp": dates, "prediction_mph": predicted, "observed_mph": observed,
                          "selected_using_test_scores": True,
                          "minimum_score_gain_percent": 100 * float(
                              candidates.set_index("location").loc[location, "gain"])})
    for method in METHODS:
        saved[f"{method}_lower_mph"] = predicted - radii[method]
        saved[f"{method}_upper_mph"] = predicted + radii[method]
    saved.to_csv(output / "traffic_prediction_interval_example.csv", index=False)
    plot_traffic_example(saved, output)


def plot_traffic_example(saved, output):
    dates = pd.to_datetime(saved.timestamp)
    widths = {method: float((saved[f"{method}_upper_mph"] - saved[f"{method}_lower_mph"]).mean())
              for method in METHODS}
    bands = {"Global": "#CDD2D9", "Local": "#AFCFE6", "Graph Smoothing": "#F8D2A7"}
    figure, axis = plt.subplots(figsize=(3.35, 2.65))
    for layer, method in enumerate(sorted(METHODS, key=widths.get, reverse=True)):
        lower, upper = saved[f"{method}_lower_mph"], saved[f"{method}_upper_mph"]
        axis.fill_between(dates, lower, upper, color=bands[method], linewidth=0, zorder=layer + 1)
    axis.plot(dates, saved.prediction_mph, color="#555555", linestyle=":", linewidth=0.5, alpha=0.4, zorder=4)
    axis.scatter(dates, saved.observed_mph, color="#303844", alpha=0.45, s=3, linewidths=0, zorder=5)
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=8, maxticks=12))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axis.tick_params(axis="x", labelsize=5.5)
    axis.set(xlim=(dates.iloc[0], dates.iloc[-1]), ylabel="Speed (mph)", xlabel="Time")
    axis.margins(x=0, y=0.05)
    handles = [Patch(facecolor=bands[method], label=f"{method} ({widths[method]:.1f} mph)") for method in METHODS]
    figure.legend(handles=handles, loc="upper center", ncol=1, frameon=False, fontsize=6.5)
    figure.tight_layout(rect=(0, 0, 1, 0.76))
    save_figure(figure, output / "traffic_prediction_interval_example.pdf")


def traffic_tables(datasets, aggregate, output):
    save_table(pd.DataFrame(datasets), output / "traffic_datasets")
    selected = aggregate[(aggregate.horizon == 30) & (aggregate.history == "24")].copy()
    selected["coverage"] *= 100
    selected = selected[["dataset", "method", "winkler_score", "coverage", "width"]]
    selected.columns = ["Dataset", "Method", "Winkler (mph)", "Coverage (%)", "Width (mph)"]
    save_table(selected.round(2), output / "traffic_interval_table")


def hospital_outputs(data, results, aggregate, histories, output):
    save_table(pd.DataFrame([{"Locations": len(data["ids"]), "Days": len(data["counts"]),
                              "Missing (%)": 100 * np.isnan(data["counts"]).mean(),
                              "Edges": int(np.count_nonzero(np.triu(data["graph"], 1)))}]),
               output / "hospital_dataset")
    results.to_csv(output / "hospital_location_results.csv", index=False)
    aggregate.to_csv(output / "hospital_results.csv", index=False)
    hospital_scores(aggregate, histories, output)


def hospital_scores(aggregate, histories, output):
    figure, axis = plt.subplots(figsize=(3.35, 2.4))
    for method in METHODS:
        values = aggregate[aggregate.method == method].set_index("history").loc[list(histories), "scaled_winkler"]
        axis.plot(range(len(histories)), values, color=COLORS[method],
                  marker=MARKERS[method], markersize=3.5, linewidth=1.2, label=method)
    axis.set_ylabel("Scaled Winkler score")
    axis.grid(axis="y", alpha=0.2)
    axis.set_xticks(range(len(histories)), ["Full" if history == "full" else history for history in histories])
    axis.set_xlabel("State's available days")
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=6.5)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(figure, output / "hospital_interval_results.pdf")


def regenerate_figures(application, output):
    from data_loading import require_data

    if application == "hospital":
        require_data(output, ["hospital_results.csv"])
        aggregate = pd.read_csv(output / "hospital_results.csv", dtype={"history": str})
        hospital_scores(aggregate, aggregate.history.drop_duplicates().tolist(), output)
    else:
        networks = ("traffic_network", "traffic_network_bay")
        require_data(output, ["traffic_results.csv", "traffic_location_results.csv",
                              "traffic_prediction_interval_example.csv"]
                     + [f"{name}_{part}.csv" for name in networks for part in ("nodes", "edges")])
        aggregate = pd.read_csv(output / "traffic_results.csv", dtype={"history": str})
        results = pd.read_csv(output / "traffic_location_results.csv", dtype={"history": str})
        histories = aggregate.history.drop_duplicates().tolist()
        horizons = sorted(aggregate.horizon.unique())
        for name in networks:
            nodes = pd.read_csv(output / f"{name}_nodes.csv", dtype={"sensor": str})
            edges = pd.read_csv(output / f"{name}_edges.csv", dtype={"source": str, "target": str})
            plot_network(nodes, edges, output, name)
        traffic_scores(aggregate, histories, horizons, output)
        sensor_improvements(results, histories, horizons, output, save_csv=False)
        plot_traffic_example(pd.read_csv(output / "traffic_prediction_interval_example.csv"), output)
    print(f"Figures regenerated from saved CSV results in {output}", flush=True)
