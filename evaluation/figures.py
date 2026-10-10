"""The figures that end up in the final report.

method_domains.png: every detector's AUC on MultiSocial against MULTITuDE
language_resource.png: fastdetect's adjusted AUC per text language, one panel per dataset
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from evaluation.auc import plain_auc, standardize
from evaluation.data import Detector, View
from evaluation.groups import RESOURCE_LEVELS, baseline_languages, baseline_trained_on, by_resource_level, \
    resource_level
from evaluation.style import INK, INK_SECONDARY, MUTED, SERIES, SURFACE, method_style, save


def draw(view: View, output: Path) -> None:
    method_domains(view, output)
    language_resource(view, output)


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
        size = 75 if marker == "^" else 42  # a triangle looks smaller than a circle or square of the same size
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
                          markersize=9 if method_style(m)[1] == "^" else 7, label=m) for m in methods]
    ax.legend(handles=handles, loc="lower left", fontsize=8)
    finish(fig, output / "method_domains.png")


def place_labels(ax, points: list[tuple[float, float, str, str]], fontsize: float = 6.5) -> None:
    """Labels next to their points, pushed apart so they don't overlap."""
    (left, right), (bottom, top) = ax.get_xlim(), ax.get_ylim()
    middle = np.median([x for x, _, _, _ in points])
    gap = 0.026 * (top - bottom)
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


# language_resource.png


def language_resource(view: View, output: Path) -> None:
    """fastdetect's adjusted AUC per text language, ordered by resource level, one panel per dataset."""
    detectors = [d for d in view.detectors if d.method == "fastdetect"]
    family = {d.key: d.model.family for d in detectors}
    table = adjusted_auc(view, ["detector", "dataset", "language"], detectors, scope="dataset").dropna(subset=["auc"])
    table["family"] = table["detector"].map(family)
    families = list(dict.fromkeys(family.values()))
    fig, axes = plt.subplots(len(view.datasets), 1, figsize=(9.5, 3.4 * len(view.datasets) + 0.9), squeeze=False)
    for ax, dataset in zip(axes[:, 0], view.datasets):
        part = table[table["dataset"] == dataset]
        languages = by_resource_level(part["language"])
        for i, name in enumerate(families):
            line = part[part["family"] == name].groupby("language")["auc"].mean().reindex(languages)
            ax.plot(range(len(languages)), line.to_numpy(), color=SERIES[i + 1], marker="osD^v"[i % 5], markersize=4,
                    linewidth=1.0, alpha=0.75, label=f"{name} (mean over sizes)", zorder=2)
        mean = part.groupby("language")["auc"].mean().reindex(languages)
        ax.plot(range(len(languages)), mean.to_numpy(), color=INK, linewidth=2.2, marker="o", markersize=5, zorder=3,
                label="mean over all fastdetect detectors")
        ax.axhline(0.5, linestyle=":", color=MUTED, linewidth=1.0, zorder=0)
        ax.set_xticks(range(len(languages)))
        ax.set_xticklabels(languages)
        ax.set_xlim(-0.6, len(languages) - 0.4)
        ax.grid(axis="x", visible=False)
        mark_resource_levels(ax, languages, {language: i for i, language in enumerate(languages)})
        ax.set_ylabel("AUC (adjusted)")
        ax.set_title(dataset_name(dataset), loc="left", fontweight="bold", pad=14)
    axes[-1, 0].set_xlabel("text language")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7)
    finish(fig, output / "language_resource.png", bottom=0.09)


def finish(fig, path: Path, bottom: float = 0.0) -> None:
    """Lays out the figure, leaving bottom free for a legend, and saves it."""
    fig.tight_layout(rect=(0, bottom, 1, 1))
    save(fig, path)


def mark_resource_levels(ax, languages: list[str], x: dict[str, int]) -> None:
    """A separator between resource levels and the level's name above its languages."""
    for level in RESOURCE_LEVELS:
        positions = [x[language] for language in languages if resource_level(language) == level]
        if not positions:
            continue
        if min(positions) > 0:
            ax.axvline(min(positions) - 0.5, color=MUTED, linewidth=0.6, linestyle="--")
        ax.text((min(positions) + max(positions)) / 2, 1.01, level, transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=7.5, color=INK_SECONDARY)
