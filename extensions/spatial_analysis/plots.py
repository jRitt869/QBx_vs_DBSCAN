"""Optional scientific plots for spatial evaluation and geographic label review."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, ScalarFormatter


def label_colors(labels: Sequence[int]) -> dict[int, object]:
    ids = sorted(set(int(label) for label in labels) - {-1})
    # Use darker colors so small areas remain legible on a white background.
    # Reserve gray for unassigned tracks; IDs also identify every area.
    palette = [matplotlib.colormaps["tab20"].colors[i] for i in (0, 2, 4, 6, 8, 10, 12, 16, 18)]
    palette += [
        matplotlib.colormaps["tab20b"].colors[i] for i in (0, 4, 8, 12, 16, 1, 5, 9, 13, 17)
    ]
    palette += list(matplotlib.colormaps["tab20c"].colors[:2])
    return {-1: "#64748b", **{label: palette[i % len(palette)] for i, label in enumerate(ids)}}


def _draw(
    ax: plt.Axes, tracks: gpd.GeoDataFrame, labels: Sequence[int], colors: Mapping[int, object]
) -> None:
    for geometry, label in zip(tracks.geometry, labels, strict=True):
        coords = np.asarray(geometry.coords)
        # Cap plot complexity; full-resolution geometry is used for label validation.
        indices = np.unique(np.linspace(0, len(coords) - 1, min(len(coords), 1000), dtype=int))
        coords = coords[indices]
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            color=colors[int(label)],
            lw=0.7 if label == -1 else 1.2,
            alpha=0.6 if label == -1 else 0.8,
            linestyle="--" if label == -1 else "-",
        )
        if np.ptp(coords, axis=0).max() == 0:
            ax.scatter(coords[0, 0], coords[0, 1], color=colors[int(label)], s=12)
    ax.set(xlabel="Longitude (degrees)", ylabel="Latitude (degrees)")
    bounds = tracks.total_bounds
    ax.set_aspect(1 / np.cos(np.deg2rad((bounds[1] + bounds[3]) / 2)))
    ax.grid(color="#cbd5e1", alpha=0.45, linewidth=0.6)
    ax.tick_params(labelsize=8)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=4))
        axis.set_major_formatter(ScalarFormatter(useOffset=False))


def plot_comparison(
    tracks: gpd.GeoDataFrame,
    truth: Sequence[int],
    predictions: Mapping[str, tuple[str, Sequence[int]]],
    destination: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(predictions) + 1,
        figsize=(3.5 * (len(predictions) + 1), 8.5),
        squeeze=False,
        layout="constrained",
    )
    panels = [
        ("Reference labels", truth),
        *[(name, assignments) for name, assignments in predictions.values()],
    ]
    bounds = tracks.total_bounds
    for ax, (title, labels) in zip(axes[0], panels, strict=True):
        _draw(ax, tracks, labels, label_colors(labels))
        ax.set_title(
            f"{title}\n{len(set(labels) - {-1})} clusters; {sum(v == -1 for v in labels)} outliers",
            fontsize=11,
        )
        width, height = max(bounds[2] - bounds[0], 0.01), max(bounds[3] - bounds[1], 0.01)
        ax.set(
            xlim=(bounds[0] - width * 0.03, bounds[2] + width * 0.03),
            ylim=(bounds[1] - height * 0.03, bounds[3] + height * 0.03),
        )
    fig.suptitle(
        "Spatial clustering comparison • colors denote membership within each panel",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def plot_metric_sweep(summary: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, algorithm, color in zip(axes, ("dbscan", "qbx"), ("#0369a1", "#7c3aed"), strict=True):
        values = summary[summary.algorithm == algorithm].sort_values("eps")
        if values.empty:
            ax.set_visible(False)
            continue
        ax.errorbar(
            values.eps,
            values.adjusted_rand_mean,
            yerr=values.adjusted_rand_std,
            marker="o",
            color=color,
            capsize=4,
            label="ARI (mean ± seed std)",
        )
        ax.plot(values.eps, values.pair_f1_mean, "s--", color="#059669", label="Pair F1")
        ax.set_xscale("log")
        ax.set(
            title=algorithm.upper(),
            xlabel="Spatial epsilon (algorithm-specific scale)",
            ylabel="Score",
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Exploratory parameter sweep on the labeled sample", fontsize=14, fontweight="bold"
    )
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_geographic_labels(
    tracks: gpd.GeoDataFrame,
    labels: Sequence[int],
    regions: Sequence[dict],
    destination: Path,
) -> None:
    colors = label_colors(labels)
    fig, (ax, legend_ax) = plt.subplots(
        2, 1, figsize=(12, 11), gridspec_kw={"height_ratios": [8, 2]}, layout="constrained"
    )
    legend_ax.set_axis_off()
    _draw(ax, tracks, labels, colors)
    handles = []
    for region in regions:
        cluster = region["cluster"]
        xmin, ymin, xmax, ymax = region["bounds"]
        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor=colors[cluster],
                lw=1.3,
            )
        )
        ax.text(
            (xmin + xmax) / 2,
            ymax + 0.025,
            str(cluster),
            fontsize=9,
            fontweight="bold",
            color=colors[cluster],
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=colors[cluster],
                lw=2,
                label=f"{cluster}: {sum(v == cluster for v in labels)} tracks",
            )
        )
    handles.append(
        Line2D(
            [0],
            [0],
            color=colors[-1],
            linestyle="--",
            label=f"-1: {sum(v == -1 for v in labels)} unassigned",
        )
    )
    legend_ax.legend(
        handles=handles,
        title="Geographic area label",
        loc="center",
        ncol=6,
        fontsize=9,
    )
    ax.set_title(
        f"Dump geographic annotations • {len(tracks)} tracks\n"
        "Colored boxes define areas; dashed gray tracks remain unassigned",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(destination, dpi=180)
    plt.close(fig)

    cols = 4
    rows = (len(regions) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(16, rows * 3.2), layout="constrained", squeeze=False
    )
    labels = np.asarray(labels)
    for ax, region in zip(axes.flat, regions, strict=False):
        cluster = region["cluster"]
        selected = tracks.loc[labels == cluster]
        _draw(ax, selected, labels[labels == cluster], colors)
        xmin, ymin, xmax, ymax = region["bounds"]
        # Expand the viewport to a square in approximate geographic distance.
        # This preserves map aspect while avoiding narrow axes and overlapping ticks.
        aspect = 1 / np.cos(np.deg2rad((ymin + ymax) / 2))
        span = max((xmax - xmin) / aspect, ymax - ymin) * 1.08
        center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
        ax.set(
            xlim=(center_x - span * aspect / 2, center_x + span * aspect / 2),
            ylim=(center_y - span / 2, center_y + span / 2),
        )
        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor=colors[cluster],
                lw=0.8,
                linestyle=":",
            )
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.set_title(f"Area {cluster} • {len(selected)} tracks", fontsize=11, fontweight="bold")
    for ax in list(axes.flat)[len(regions) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Regional label review • dotted boxes define areas; each panel has its own scale",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(destination.with_name(destination.stem + "_details.png"), dpi=150)
    plt.close(fig)
