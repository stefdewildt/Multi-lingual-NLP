"""The AUC, plain and adjusted for confounders by direct standardization.

Text language, text length and generator all affect how well a text can be
detected, and the groups we compare (detectors, datasets, languages) differ
in how their texts are spread over them. The adjusted AUC therefore splits
the texts into subgroups of these variables, computes the AUC within each
subgroup (the machine texts of one generator against the human texts of the
same language and length bin) and averages them with equal weight over one
set of subgroups C shared by every group in a comparison:

AUC_adj = 1/|C| * sum over c in C of AUC(machine texts in c vs human texts in c)

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LENGTH_EDGES = [0, 16, 32, 64, 128, 256, 512, float("inf")]
LENGTH_LABELS = ["<16", "16-32", "32-64", "64-128", "128-256", "256-512", "512+"]
"""Text-length bins in mT5 tokens, the same for every dataset. Doubling
widths, since detection improves roughly with log length."""

SUBGROUP_VARIABLES = ["language", "length", "generator"]

MIN_HUMAN = 20
MIN_MACHINE = 10
"""A subgroup only counts if it has at least this many human texts (of its
language and length bin) and machine texts (of its generator). We chose these
numbers because you want these to be as small as possible too loose as little 
data as possible, but  variance/noise get's too much if you allow smaller 
subgroups, so this felt right."""

MIN_SHARE = 0.9
"""A subgroup is part of C if at least this share of the compared groups has it."""

MIN_COVERAGE = 0.75
"""A group missing more than 1 - MIN_COVERAGE of C gets no value."""


def length_bin(lengths: np.ndarray | pd.Series) -> pd.Categorical:
    """Which LENGTH_LABELS bin each length falls in."""
    return pd.cut(np.asarray(lengths, dtype=float), LENGTH_EDGES, labels=LENGTH_LABELS, right=False)


def plain_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """The AUC of all texts together, NaN without both human and machine texts."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def hanley_mcneil_se(area: float, n_machine: float, n_human: float) -> float:
    """Standard error of an AUC (Hanley & McNeil, 1982)."""
    q1 = area / (2 - area)
    q2 = 2 * area**2 / (1 + area)
    variance = (area * (1 - area) + (n_machine - 1) * (q1 - area**2) + (n_human - 1) * (q2 - area**2)) / (
        n_machine * n_human)
    return float(np.sqrt(max(variance, 0.0)))


def sorted_auc(sorted_human: np.ndarray, machine: np.ndarray) -> float:
    """Mann-Whitney AUC against sorted human scores, ties count half."""
    below = np.searchsorted(sorted_human, machine, side="left")
    below_or_equal = np.searchsorted(sorted_human, machine, side="right")
    return float((below + below_or_equal).sum() / 2 / (len(sorted_human) * len(machine)))


def subgroup_aucs(frame: pd.DataFrame) -> pd.DataFrame:
    """The AUC per (language, length, generator) subgroup, against the human texts
    of the same language and length. Too small subgroups are left out."""
    texts = pd.DataFrame({
        "language": frame["language"],
        "length": length_bin(frame["computed_length"]),
        "generator": frame["multi_label"],
        "score": frame["score"].to_numpy(),
    })
    is_human = (frame["label"] == 0).to_numpy()
    humans = {
        key: np.sort(group["score"].to_numpy())
        for key, group in texts[is_human].groupby(["language", "length"], observed=True)
    }
    rows = []
    for (language, length, generator), group in texts[~is_human].groupby(SUBGROUP_VARIABLES, observed=True):
        human = humans.get((language, length))
        machine = group["score"].to_numpy()
        if human is None or len(human) < MIN_HUMAN or len(machine) < MIN_MACHINE:
            continue
        area = sorted_auc(human, machine)
        rows.append({
            "language": str(language), "length": str(length), "generator": str(generator), "auc": area,
            "auc_se": hanley_mcneil_se(area, len(machine), len(human)),
            "n_human": len(human), "n_machine": len(machine),
        })
    return pd.DataFrame(rows, columns=SUBGROUP_VARIABLES + ["auc", "auc_se", "n_human", "n_machine"])


def shared_subgroups(present: pd.DataFrame, by: list[str], over: list[str]) -> pd.DataFrame:
    """C: the subgroups that at least MIN_SHARE of the groups have. Very sparse groups are ignored."""
    n_groups = len(present[by].drop_duplicates())
    counts = present.groupby(over).size()
    common = counts[counts.to_numpy() >= 0.5 * n_groups].reset_index()[over]
    per_group = present.merge(common, on=over).groupby(by).size()
    viable = per_group[per_group.to_numpy() >= 0.5 * len(common)].reset_index()[by]
    counts = present.merge(viable, on=by).groupby(over).size()
    return counts[counts.to_numpy() >= MIN_SHARE * max(len(viable), 1)].reset_index()[over]


def filled_means(used: pd.DataFrame, by: list[str], over: list[str]) -> pd.Series:
    """Each group's mean AUC over C. Missing subgroups come from an additive group + subgroup model."""
    wide = used.set_index(by + over)["auc"].unstack(over)
    group_effect = wide.mean(axis=1)
    subgroup_effect = pd.Series(0.0, index=wide.columns)
    for _ in range(20):
        subgroup_effect = wide.sub(group_effect, axis=0).mean(axis=0)
        subgroup_effect -= subgroup_effect.mean()
        group_effect = wide.sub(subgroup_effect, axis=1).mean(axis=1)
    fitted = pd.DataFrame(np.add.outer(group_effect.to_numpy(), subgroup_effect.to_numpy()), index=wide.index,
                          columns=wide.columns)
    return wide.where(wide.notna(), fitted).mean(axis=1).rename("auc")


def standardize(cells: pd.DataFrame, by: list[str], scope: str = "dataset") -> tuple[pd.DataFrame, pd.DataFrame]:
    """The adjusted AUC per group of by. scope "dataset" picks C per dataset,
    "shared" once for both. Returns (table, C)."""
    over = [v for v in SUBGROUP_VARIABLES if v not in by]
    columns = by + ["auc", "auc_se", "n_cells", "complete", "n_human", "n_machine"]
    if cells.empty or not over:
        return pd.DataFrame(columns=columns), pd.DataFrame()
    parts = [("", cells)] if scope == "shared" else list(cells.groupby("dataset"))
    tables, supports = [], []
    for dataset, part in parts:
        groups = part[by].drop_duplicates()
        support = shared_subgroups(part.drop_duplicates(by + over)[by + over], by, over)
        used = part.merge(support, on=over)
        # the human texts of one language and length bin are shared by its generators, count them once
        human_key = by + [v for v in ("language", "length") if v in over]
        table = used.groupby(by).agg(
            n_cells=("auc", "size"), se2=("auc_se", lambda v: (v**2).sum()), n_machine=("n_machine", "sum"))
        table["n_human"] = used.drop_duplicates(human_key).groupby(by)["n_human"].sum()
        table = groups.merge(table.reset_index(), on=by, how="left")
        table["n_cells"] = table["n_cells"].fillna(0).astype(int)
        table["complete"] = table["n_cells"] == len(support)
        table["auc_se"] = np.sqrt(table["se2"]) / table["n_cells"].where(table["n_cells"] > 0)
        table = table.merge(filled_means(used, by, over).reset_index(), on=by, how="left") if len(support) else \
            table.assign(auc=np.nan)
        table.loc[(table["n_cells"] < MIN_COVERAGE * len(support)) | (len(support) == 0), ["auc", "auc_se"]] = np.nan
        tables.append(table[columns])
        supports.append(support.assign(dataset=dataset) if scope == "dataset" else support)
    return pd.concat(tables, ignore_index=True), pd.concat(supports, ignore_index=True)
