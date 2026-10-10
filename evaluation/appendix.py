"""The report's appendix figures.

method_{length,language,generator}.png: AUC per detector and text length / language / generator
language_baselines.png: baselines on their own training languages and on the others
language_resource.png: fastdetect's adjusted AUC per text language, one panel per dataset
language_ab_test.png: an English-only against a multilingual scoring model of the same size
size.png: fastdetect against the size of its scoring model
"""

from __future__ import annotations

from pathlib import Path

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from evaluation.auc import LENGTH_LABELS
from evaluation.data import Detector, View
from evaluation.figures import FAMILIES, adjusted_auc, dataset_name, finish, heatmap, mark_resource_levels, \
    resource_level_gaps, short_name
from evaluation.groups import baseline_languages, baseline_trained_on, baseline_training_languages, by_resource_level
from evaluation.style import INK, INK_SECONDARY, MUTED, SERIES, SURFACE, method_style

GENERATORS = ["gpt-3.5-turbo-0125", "gemini", "Llama-2-70b-chat-hf", "vicuna-13b", "Mistral-7B-Instruct-v0.2",
              "aya-101", "opt-iml-max-30b", "v5-Eagle-7B-HF"]
"""The models that wrote the machine texts, in the order the figures show them."""

def draw(view: View, output: Path) -> None:
    for variable in ("length", "language", "generator"):
        method_heatmap(view, variable, output)
    language_resource(view, output)
    language_baselines(view, output)
    language_ab_test(view, output)
    size(view, output)


def generator_name(generator: str) -> str:
    match generator:
        case "gpt-3.5-turbo-0125":
            return "GPT-3.5"
        case "gemini":
            return "Gemini"
        case "Llama-2-70b-chat-hf":
            return "Llama-2-70B"
        case "vicuna-13b":
            return "Vicuna-13B"
        case "Mistral-7B-Instruct-v0.2":
            return "Mistral-7B"
        case "aya-101":
            return "Aya-101"
        case "opt-iml-max-30b":
            return "OPT-IML-30B"
        case "v5-Eagle-7B-HF":
            return "Eagle-7B"
        case _:
            return generator


def dataset_style(dataset: str) -> tuple[str, Literal["-", "--"]]:
    """(marker, line style) of a dataset in line charts."""
    match dataset:
        case "multisocial":
            return "o", "-"
        case _:
            return "s", "--"


# method_<variable>.png


def best_detectors(view: View, dataset: str) -> list[Detector]:
    """The best size per fastdetect family, all detectgpt detectors and the all-language baselines."""
    scored = [d for d in view.detectors if dataset in view.frames[d]]
    fast = [d for d in scored if d.method == "fastdetect"]
    overall = adjusted_auc(view, ["detector", "dataset"], fast, scope="dataset")
    auc = overall[overall["dataset"] == dataset].set_index("detector")["auc"]
    best = []
    for family in FAMILIES:
        sizes = [d for d in fast if d.model.family == family]
        if sizes:
            best.append(max(sizes, key=lambda d: auc.get(d.key, -1)))
    return (best + [d for d in scored if d.method == "detectgpt"]
            + [d for d in scored if d.method == "baseline" and baseline_languages(d.config) == "all"])


def method_heatmap(view: View, variable: str, output: Path) -> None:
    """Adjusted AUC per detector and length, language or generator, one panel per dataset."""
    match variable:
        case "length":
            name = "text length (mT5 tokens)"
        case "language":
            name = "text language"
        case _:
            name = "generator"
    panels = []
    for dataset in view.datasets:
        detectors = best_detectors(view, dataset)
        table = adjusted_auc(view, ["detector", "dataset", variable], detectors, scope="dataset")
        table = table[table["dataset"] == dataset].dropna(subset=["auc"])
        present = set(table[variable])
        match variable:
            case "length":
                columns = [b for b in LENGTH_LABELS if b in present]
            case "language":
                columns = by_resource_level(present)
            case _:
                columns = [g for g in GENERATORS if g in present]
        grid = table.pivot(index="detector", columns=variable, values="auc").reindex(
            index=[d.key for d in detectors], columns=columns)
        if variable == "generator":
            grid.columns = [generator_name(g) for g in columns]
        panels.append((dataset, detectors, grid))
    width = max(6.0, 2.6 + 0.5 * max(grid.shape[1] for _, _, grid in panels))
    heights = [len(detectors) for _, detectors, _ in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(width, 1.6 + 0.32 * sum(heights) + 0.7 * len(panels)),
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    for ax, (dataset, detectors, grid) in zip(axes[:, 0], panels):
        gaps = [i for i in range(1, len(detectors)) if detectors[i].method != detectors[i - 1].method]
        columns = list(grid.columns)
        x = heatmap(ax, grid, [short_name(d) for d in detectors], [method_style(d.method)[0] for d in detectors],
                    gaps=gaps, column_gaps=resource_level_gaps(columns) if variable == "language" else ())
        if variable == "language":
            mark_resource_levels(ax, columns, dict(zip(columns, x)), separators=False)
        ax.set_title(dataset_name(dataset), loc="left", fontweight="bold", pad=14 if variable == "language" else 6)
    axes[-1, 0].set_xlabel(name)
    finish(fig, output / f"method_{variable}.png")


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


# language_baselines.png


def baseline_label(detector: Detector) -> str:
    match baseline_languages(detector.config):
        case "en":
            return f"{short_name(detector)}, en"
        case "big_eu":
            return f"{short_name(detector)}, big EU"
        case "less_common":
            return f"{short_name(detector)}, less common"
        case _:
            return short_name(detector)


def language_baselines(view: View, output: Path) -> None:
    """Each baseline's adjusted AUC on its own training languages and on the others."""
    order = ["multisocial", "multitude", "both"], ["all", "en", "big_eu", "less_common"]
    baselines = sorted((d for d in view.detectors if d.method == "baseline"),
                       key=lambda d: (order[0].index(baseline_trained_on(d.config)),
                                      order[1].index(baseline_languages(d.config))))
    gaps = [i for i in range(1, len(baselines))
            if baseline_trained_on(baselines[i].config) != baseline_trained_on(baselines[i - 1].config)]
    figsize = (2.6 + 2.0 * len(view.datasets), 1.8 + 0.3 * len(baselines))
    fig, axes = plt.subplots(1, len(view.datasets), figsize=figsize, sharey=True, squeeze=False)
    for ax, dataset in zip(axes[0], view.datasets):
        per_language = adjusted_auc(view, ["detector", "dataset", "language"], baselines, scope="dataset")
        per_language = per_language[per_language["dataset"] == dataset].set_index(["detector", "language"])["auc"]
        overall = adjusted_auc(view, ["detector", "dataset"], baselines, scope="dataset")
        overall = overall[overall["dataset"] == dataset].set_index("detector")["auc"]
        grid = pd.DataFrame(index=[d.key for d in baselines], columns=["own", "other"], dtype=float)
        for d in baselines:
            trained = baseline_training_languages(d.config)
            if trained is None:
                grid.loc[d.key, "own"] = overall.get(d.key, np.nan)
            elif d.key in per_language.index.get_level_values(0):
                aucs = per_language.loc[d.key]
                grid.loc[d.key, "own"] = aucs[aucs.index.isin(trained)].mean()
                grid.loc[d.key, "other"] = aucs[~aucs.index.isin(trained)].mean()
        heatmap(ax, grid, [baseline_label(d) for d in baselines], [INK_SECONDARY] * len(baselines), gaps=gaps)
        ax.set_title(dataset_name(dataset), fontweight="bold")
    finish(fig, output / "language_baselines.png")


# language_ab_test.png


def language_ab_test(view: View, output: Path) -> None:
    """SmolLM2-1.7B (English only) against EuroLLM-1.7B (35 languages) per text language."""
    names = {"SmolLM2-1.7B": "SmolLM2-1.7B, English only", "EuroLLM-1.7B": "EuroLLM-1.7B, 35 languages"}
    pair = [d for d in view.detectors if d.method == "fastdetect" and short_name(d) in names]
    table = adjusted_auc(view, ["detector", "dataset", "language"], pair, scope="shared").dropna(subset=["auc"])
    languages = by_resource_level(table["language"])
    x = {language: i for i, language in enumerate(languages)}
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for d, color in zip(pair, (SERIES[1], SERIES[2])):
        for dataset in view.datasets:
            part = table[(table["detector"] == d.key) & (table["dataset"] == dataset)]
            part = part.assign(x=part["language"].map(x)).sort_values("x")
            marker, line = dataset_style(dataset)
            offset = -0.12 if dataset == "multisocial" else 0.12
            ax.errorbar(part["x"] + offset, part["auc"], yerr=1.96 * part["auc_se"], color=color, marker=marker,
                        linestyle=line, linewidth=1.3, markersize=5, markeredgecolor=SURFACE, elinewidth=0.8, capsize=2)
    ax.set_xticks(range(len(languages)))
    ax.set_xticklabels(languages)
    ax.set_xlim(-0.6, len(languages) - 0.4)
    ax.grid(axis="x", visible=False)
    mark_resource_levels(ax, languages, x)
    ax.set_xlabel("text language")
    ax.set_ylabel("AUC (adjusted)")
    handles = [Line2D([0], [0], color=c, linewidth=2, label=names[short_name(d)])
               for d, c in zip(pair, (SERIES[1], SERIES[2]))]
    handles += [Line2D([0], [0], color=INK_SECONDARY, marker=dataset_style(ds)[0], linestyle=dataset_style(ds)[1],
                           label=dataset_name(ds)) for ds in view.datasets]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)
    finish(fig, output / "language_ab_test.png")


# size.png


def size(view: View, output: Path) -> None:
    """fastdetect's adjusted AUC against scoring model size, per model family."""
    fast = [d for d in view.detectors if d.method == "fastdetect"]
    sizes = pd.Series([d.model.family for d in fast]).value_counts()
    detectors = [d for d in fast if sizes[d.model.family] > 1]
    table = adjusted_auc(view, ["detector", "dataset"], detectors, scope="shared").dropna(subset=["auc"])
    model = {d.key: d.model for d in detectors}
    table["family"] = table["detector"].map({key: m.family for key, m in model.items()})
    table["params"] = table["detector"].map({key: m.params for key, m in model.items()})
    families = [f for f in FAMILIES if f in set(table["family"])]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for i, family in enumerate(families):
        for dataset in view.datasets:
            part = table[(table["family"] == family) & (table["dataset"] == dataset)].sort_values("params")
            marker, line = dataset_style(dataset)
            ax.errorbar(part["params"], part["auc"], yerr=1.96 * part["auc_se"], color=SERIES[i], marker=marker,
                        linestyle=line, linewidth=1.3, markersize=5, markeredgecolor=SURFACE, capsize=2, elinewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.3, 1, 3])
    ax.set_xticklabels(["0.1B", "0.3B", "1B", "3B"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(min(0.6, float(table["auc"].min()) - 0.08), None)  # room for the legend under the lines
    ax.set_xlabel("scoring model parameters (log scale)")
    ax.set_ylabel("AUC (adjusted)")
    ax.grid(axis="x", visible=False)
    handles = [Line2D([0], [0], color=SERIES[i], linewidth=2, label=f) for i, f in enumerate(families)]
    handles += [Line2D([0], [0], color=INK_SECONDARY, marker=dataset_style(ds)[0], linestyle=dataset_style(ds)[1],
                           label=dataset_name(ds)) for ds in view.datasets]
    ax.legend(handles=handles, loc="lower right", ncol=2)
    finish(fig, output / "size.png")
