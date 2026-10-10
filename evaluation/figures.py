"""The figures that end up in the final report.

method_domains.png: every detector's AUC on MultiSocial against MULTITuDE
language_scoring_model.png: fastdetect per scoring model family and text language
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from evaluation.auc import plain_auc, standardize
from evaluation.data import Detector, View
from evaluation.groups import RESOURCE_LEVELS, baseline_languages, baseline_trained_on, by_resource_level, \
    resource_level
from evaluation.style import AUC_CMAP, INK, INK_SECONDARY, MUTED, SURFACE, method_style, save


FAMILIES = ["Qwen", "EuroLLM", "Llama", "SmolLM2", "GPT-2"]
"""The scoring model families, from most to fewest training languages (119, 35, 8, English only)."""

COLUMN_GAP = 0.3
"""The whitespace between groups of heatmap columns, in cells."""


def draw(view: View, output: Path) -> None:
    method_domains(view, output)
    language_scoring_model(view, output)


def dataset_name(dataset: str) -> str:
    match dataset:
        case "multisocial":
            return "MultiSocial"
        case "multitude":
            return "MULTITuDE"
        case _:
            return dataset


def short_name(detector: Detector) -> str:
    """The scoring model, or for a baseline its training dataset(s)."""
    if detector.method == "baseline":
        match baseline_trained_on(detector.config):
            case "multisocial":
                return "baseline MS"
            case "multitude":
                return "baseline MT"
            case _:
                return "baseline MS+MT"
    return (detector.config.scoring or "").split("/")[-1].removesuffix("-Instruct")


def adjusted_auc(view: View, by: list[str], detectors: list[Detector], scope: str) -> pd.DataFrame:
    """The adjusted AUC per group of by, comparing only these detectors."""
    cells = view.cells[view.cells["detector"].isin({d.key for d in detectors})]
    table, _ = standardize(cells, by, scope)
    return table


def heatmap(ax, grid: pd.DataFrame, labels: list[str], colors: list[str], boxed: pd.DataFrame | None = None,
            gaps: Sequence[int] = (), column_gaps: Sequence[int] = ()) -> list[float]:
    """An annotated AUC heatmap. gaps adds a white line above those rows, column_gaps
    whitespace left of those columns, boxed frames cells. Returns the columns' x positions."""
    values = grid.to_numpy(dtype=float)
    rows, columns = values.shape
    x = [j + COLUMN_GAP * sum(g <= j for g in column_gaps) for j in range(columns)]
    norm = TwoSlopeNorm(vmin=0.3, vcenter=0.5, vmax=1.0)
    starts = [0, *column_gaps, columns]
    for start, end in zip(starts, starts[1:]):  # one block of cells between every two gaps
        ax.pcolormesh(np.arange(end - start + 1) + x[start] - 0.5, np.arange(rows + 1) - 0.5, values[:, start:end],
                      cmap=AUC_CMAP, norm=norm, edgecolors="none")
    ax.set_xlim(-0.5, x[-1] + 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.grid(False)
    for i, j in np.argwhere(~np.isnan(values)):
        ax.text(x[j], i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if values[i, j] > 0.8 or values[i, j] < 0.38 else INK)
    if boxed is not None:
        for i, j in np.argwhere(boxed.to_numpy(dtype=bool)):
            ax.add_patch(Rectangle((x[j] - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.6))
    for i in gaps:
        ax.axhline(i - 0.5, color=SURFACE, linewidth=5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for tick, color in zip(ax.get_yticklabels(), colors):
        tick.set_color(color)
    ax.set_xticks(x)
    ax.set_xticklabels(grid.columns)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return x


def resource_level_gaps(languages: list[str]) -> list[int]:
    """The positions where the next resource level starts."""
    return [i for i in range(1, len(languages)) if resource_level(languages[i]) != resource_level(languages[i - 1])]


# method_domains.png


def method_domains(view: View, output: Path) -> None:
    """The plain AUC of every detector on both datasets. Not adjusted, since that
    would drop the baselines trained on MULTITuDE."""
    detectors = [d for d in view.detectors if set(view.datasets) <= set(view.frames[d])
                 and (d.method != "baseline" or baseline_languages(d.config) == "all")]
    auc = {(d, ds): plain_auc(f["label"].to_numpy(), f["score"].to_numpy())
           for d in detectors for ds, f in view.frames[d].items()}
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    points = []
    for d in detectors:
        color, marker = method_style(d.method)
        x, y = auc[(d, "multitude")], auc[(d, "multisocial")]
        size = 125 if marker == "^" else 70  # a triangle looks smaller than a circle or square of the same size
        ax.scatter(x, y, color=color, marker=marker, s=size, edgecolors=SURFACE, linewidths=0.8, zorder=3)
        points.append((x, y, short_name(d), color))
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    ax.plot([0, 1], [0, 1], linestyle=":", color=MUTED, linewidth=1.0, zorder=1)
    ax.set_xlim(min(xs) - 0.08, max(xs) + 0.08)  # room for the labels next to the outer points
    ax.set_ylim(min(ys) - 0.04, max(ys) + 0.04)
    place_labels(ax, points)
    ax.set_xlabel("AUC on MULTITuDE (news)")
    ax.set_ylabel("AUC on MultiSocial (social media)")
    methods = list(dict.fromkeys(d.method for d in detectors))
    handles = [Line2D([0], [0], color=method_style(m)[0], marker=method_style(m)[1], linestyle="none",
                          markersize=11.5 if method_style(m)[1] == "^" else 9, label=m) for m in methods]
    ax.legend(handles=handles, loc="lower left", fontsize=9.5)
    finish(fig, output / "method_domains.png")


def place_labels(ax, points: list[tuple[float, float, str, str]], fontsize: float = 8.75) -> None:
    """Labels next to their points, pushed apart so they don't overlap."""
    (left, right), (bottom, top) = ax.get_xlim(), ax.get_ylim()
    middle = np.median([x for x, _, _, _ in points])
    height = ax.get_window_extent().height * 72 / ax.figure.dpi  # the axes height in points
    gap = 1.3 * fontsize / height * (top - bottom)  # one line of text, so the labels never overlap
    for side in (-1, 1):
        placed: list[float] = []
        chosen = sorted((p for p in points if (p[0] < middle) == (side < 0)), key=lambda p: p[1])
        for x, y, text, color in chosen:
            y_text = max(y, placed[-1] + gap) if placed else y
            placed.append(y_text)
            x_text = x + side * 0.035 * (right - left)
            ax.annotate(text, (x, y), xytext=(x_text, y_text), textcoords="data", fontsize=fontsize, va="center",
                        ha="left" if side > 0 else "right", color=color,
                        arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.4, "alpha": 0.6})


# language_scoring_model.png


def language_scoring_model(view: View, output: Path) -> None:
    """fastdetect's adjusted AUC per model family and language. Training languages are framed."""
    fast = [d for d in view.detectors if d.method == "fastdetect"]
    model = {d.key: d.model for d in fast}
    training = {m.family: m.training_languages for m in model.values()}
    table = adjusted_auc(view, ["detector", "dataset", "language"], fast, scope="dataset").dropna(subset=["auc"])
    table["family"] = table["detector"].map({key: m.family for key, m in model.items()})
    sizes = table.groupby("family")["detector"].nunique()
    fig, axes = plt.subplots(len(view.datasets), 1, figsize=(11, 0.9 + 2.5 * len(view.datasets)), squeeze=False)
    for ax, dataset in zip(axes[:, 0], view.datasets):
        part = table[table["dataset"] == dataset]
        families = [f for f in FAMILIES if f in set(part["family"])]
        languages = by_resource_level(part["language"])
        grid = part.groupby(["family", "language"])["auc"].mean().unstack().reindex(index=families, columns=languages)
        boxed = pd.DataFrame([[language in training[f] for language in languages] for f in families],
                             index=families, columns=languages)
        labels = [f"{f} (mean of sizes)" if sizes[f] > 1 else f for f in families]
        x = heatmap(ax, grid, labels, [INK_SECONDARY] * len(families), boxed=boxed,
                    column_gaps=resource_level_gaps(languages))
        mark_resource_levels(ax, languages, dict(zip(languages, x)), separators=False)
        ax.set_title(dataset_name(dataset), loc="left", fontweight="bold", pad=14)
    axes[-1, 0].set_xlabel("text language")
    finish(fig, output / "language_scoring_model.png")


def finish(fig, path: Path, bottom: float = 0.0) -> None:
    """Lays out the figure, leaving bottom free for a legend, and saves it."""
    fig.tight_layout(rect=(0, bottom, 1, 1))
    save(fig, path)


def mark_resource_levels(ax, languages: list[str], x: dict[str, float], separators: bool = True) -> None:
    """A separator between resource levels (unless separators is False) and the level's name above its languages."""
    for level in RESOURCE_LEVELS:
        positions = [x[language] for language in languages if resource_level(language) == level]
        if not positions:
            continue
        if separators and min(positions) > 0:
            ax.axvline(min(positions) - 0.5, color=MUTED, linewidth=0.6, linestyle="--")
        ax.text((min(positions) + max(positions)) / 2, 1.01, level, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=7.5, color=INK_SECONDARY)
