import argparse
import csv
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


AB_ORDER = [
    "full",
    "no_llm_semantic",
    "no_disentangle_transfer",
    "no_city_group_pref",
    "no_generator_constraints",
]

AB_LABEL = {
    "full": "Full",
    "no_llm_semantic": "w/o LLM Semantic",
    "no_disentangle_transfer": "w/o Disentangle+Transfer",
    "no_city_group_pref": "w/o City Group Pref",
    "no_generator_constraints": "w/o Generator Spatial Ctx",
}

METRICS = [
    ("full_f1", "Full-F1"),
    ("full_pairs_f1", "Full-Pairs-F1"),
    ("rep", "REP"),
]

HP_PATTERNS = {
    "hparam_lambda_pair": r"_lambda_pair_x([0-9.]+)_s\d+$",
    "hparam_transition_strength": r"_trans_strength_x([0-9.]+)_s\d+$",
    "hparam_eta_fixed": r"_eta_fixed_([0-9.]+)_s\d+$",
    "hparam_seq_num_layers": r"_seq_num_layers_([0-9.]+)_s\d+$",
}

HP_LABELS = {
    "hparam_lambda_pair": "Lambda Pair Multiplier",
    "hparam_transition_strength": "Transition Strength Multiplier",
    "hparam_eta_fixed": "Eta Fixed",
    "hparam_seq_num_layers": "Seq Num Layers",
}


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "legend.fontsize": 11,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.facecolor": "#f2f2f2",
            "figure.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 1.1,
            "grid.color": "#c9c9c9",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
        }
    )


def _to_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (ValueError, TypeError):
        return float("nan")


def _read_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("status", "")).lower() != "ok":
                continue
            picked_split = str(row.get("picked_split", "")).upper()
            if picked_split and picked_split != "TEST":
                continue
            rows.append(row)
    return rows


def _parse_ablation_key(exp_name: str) -> str:
    m = re.search(r"_ablation_(.+?)_s\d+$", exp_name)
    return m.group(1) if m else exp_name


def _ablation_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for r in rows:
        if r.get("group") != "ablation":
            continue
        key = _parse_ablation_key(str(r.get("exp_name", "")))
        if key == "hard_constraint_variant":
            continue
        if key not in AB_ORDER:
            continue
        rr = dict(r)
        rr["ab_key"] = key
        rr["ab_label"] = AB_LABEL.get(key, key)
        rr["ab_order"] = AB_ORDER.index(key)
        out.append(rr)

    out.sort(key=lambda r: int(r["ab_order"]))
    return out


def _ablation_metric_map(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    ab_rows = _ablation_rows(rows)
    out: Dict[str, Dict[str, float]] = {}
    for r in ab_rows:
        key = str(r["ab_key"])
        out[key] = {
            "full_f1": _to_float(r, "full_f1"),
            "full_pairs_f1": _to_float(r, "full_pairs_f1"),
            "rep": _to_float(r, "rep"),
        }
    return out


def plot_ablation_grouped_by_metric(rows: List[Dict[str, str]], out_dir: str, prefix: str) -> str:
    ab_rows = _ablation_rows(rows)
    if not ab_rows:
        raise RuntimeError("No ablation rows found (after excluding hard_constraint_variant).")

    variants = [r["ab_key"] for r in ab_rows]
    labels = [AB_LABEL[v] for v in variants]

    x = np.arange(len(METRICS), dtype=float)
    n_var = len(variants)
    width = 0.14 if n_var >= 5 else 0.18
    offsets = (np.arange(n_var) - (n_var - 1) / 2.0) * width

    colors = ["#b8cbe0", "#8fb6d8", "#6fa3cf", "#4f8fc4", "#2f79b8"]
    hatches = ["//", "\\\\", "..", "xx", "--"]

    fig, ax = plt.subplots(figsize=(11.8, 6.4), constrained_layout=True)
    ax.grid(True, axis="y", alpha=0.8)

    for i, row in enumerate(ab_rows):
        vals = np.array([_to_float(row, m[0]) for m in METRICS], dtype=float)
        bars = ax.bar(
            x + offsets[i],
            vals,
            width=width,
            color=colors[i % len(colors)],
            edgecolor="#2d2d2d",
            linewidth=1.0,
            hatch=hatches[i % len(hatches)],
            label=labels[i],
        )
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 0.002,
                f"{h:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in METRICS])
    ax.set_ylabel("Score")
    ax.set_xlabel("Metrics")
    ax.set_title("Ablation Study")
    ax.legend(loc="upper right", ncol=2, frameon=True)

    out_path = os.path.join(out_dir, f"{prefix}_ablation_grouped_metrics.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_ablation_metric_split_two_datasets(
    rows_a: List[Dict[str, str]],
    rows_b: List[Dict[str, str]],
    label_a: str,
    label_b: str,
    out_dir: str,
    prefix: str,
) -> List[str]:
    map_a = _ablation_metric_map(rows_a)
    map_b = _ablation_metric_map(rows_b)

    variants = [k for k in AB_ORDER if k in map_a and k in map_b]
    if not variants:
        raise RuntimeError("No common ablation variants found between two datasets.")

    x = np.arange(len(variants), dtype=float)
    width = 0.34
    saved: List[str] = []

    for metric_key, metric_label in METRICS:
        y_a = np.array([map_a[v][metric_key] for v in variants], dtype=float)
        y_b = np.array([map_b[v][metric_key] for v in variants], dtype=float)

        fig, ax = plt.subplots(figsize=(11.6, 6.2), constrained_layout=True)
        ax.grid(True, axis="y", alpha=0.8)

        bars_a = ax.bar(
            x - width / 2,
            y_a,
            width=width,
            color="#8fb6d8",
            edgecolor="#2d2d2d",
            linewidth=1.0,
            hatch="//",
            label=label_a,
        )
        bars_b = ax.bar(
            x + width / 2,
            y_b,
            width=width,
            color="#e29578",
            edgecolor="#2d2d2d",
            linewidth=1.0,
            hatch="\\\\",
            label=label_b,
        )

        for b in list(bars_a) + list(bars_b):
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + max(0.001, 0.01 * max(np.nanmax(y_a), np.nanmax(y_b))),
                f"{h:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([AB_LABEL[v] for v in variants], rotation=16, ha="right")
        ax.set_xlabel("Ablation Variants")
        ax.set_ylabel(metric_label)
        ax.set_title(f"Ablation Comparison by {metric_label}")
        ax.legend(loc="best", frameon=True)

        out_path = os.path.join(out_dir, f"{prefix}_ablation_{metric_key}_two_datasets.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        saved.append(out_path)

    return saved


def _extract_hparam_rows(rows: List[Dict[str, str]], group: str) -> List[Dict[str, str]]:
    pat = HP_PATTERNS[group]
    out: List[Dict[str, str]] = []
    for r in rows:
        if r.get("group") != group:
            continue
        m = re.search(pat, str(r.get("exp_name", "")))
        if m is None:
            continue
        rr = dict(r)
        rr["x"] = float(m.group(1))
        out.append(rr)

    out.sort(key=lambda r: float(r["x"]))
    return out


def plot_single_hparam_dual_axis(
    rows: List[Dict[str, str]],
    group: str,
    out_dir: str,
    prefix: str,
) -> str:
    hp = _extract_hparam_rows(rows, group)
    if not hp:
        raise RuntimeError(f"No rows found for {group}.")

    x = np.array([float(r["x"]) for r in hp], dtype=float)
    y_full_f1 = np.array([_to_float(r, "full_f1") for r in hp], dtype=float)
    y_full_pairs = np.array([_to_float(r, "full_pairs_f1") for r in hp], dtype=float)
    y_rep = np.array([_to_float(r, "rep") for r in hp], dtype=float)

    fig, ax1 = plt.subplots(figsize=(10.6, 6.2), constrained_layout=True)
    ax1.grid(True, alpha=0.8)

    l1 = ax1.plot(
        x,
        y_full_f1,
        marker="o",
        color="#3b7fb6",
        linewidth=2.2,
        markersize=7,
        label="Full-F1",
    )
    l2 = ax1.plot(
        x,
        y_full_pairs,
        marker="s",
        color="#e6614c",
        linewidth=2.2,
        markersize=7,
        label="Full-Pairs-F1",
    )
    ax1.set_xlabel(HP_LABELS[group])
    ax1.set_ylabel("Full-F1 / Full-Pairs-F1")

    ax2 = ax1.twinx()
    l3 = ax2.plot(
        x,
        y_rep,
        marker="^",
        color="#4aa786",
        linewidth=2.2,
        markersize=7,
        label="REP",
    )
    ax2.set_ylabel("REP")

    for xi, yi in zip(x, y_full_f1):
        ax1.text(xi, yi, f"{yi:.4f}", fontsize=9, ha="center", va="bottom")
    for xi, yi in zip(x, y_full_pairs):
        ax1.text(xi, yi, f"{yi:.4f}", fontsize=9, ha="center", va="bottom")
    for xi, yi in zip(x, y_rep):
        ax2.text(xi, yi, f"{yi:.4f}", fontsize=9, ha="center", va="bottom")

    lines = l1 + l2 + l3
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="best", frameon=True)
    ax1.set_title(f"Hyperparameter Analysis: {HP_LABELS[group]}")

    out_path = os.path.join(out_dir, f"{prefix}_{group}_dual_axis.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ablation and hyperparameter figures from summary CSV.")
    parser.add_argument(
        "--csv",
        type=str,
        default="./pipeline_runs/yelp_soft_pipeline_v1/ablation/batch_output/summary_yelp_soft_pipeline_v1_ablation_latest.csv",
    )
    parser.add_argument("--csv_b", type=str, default="")
    parser.add_argument("--dataset_a_label", type=str, default="Dataset-A")
    parser.add_argument("--dataset_b_label", type=str, default="Dataset-B")
    parser.add_argument("--out_dir", type=str, default="./pipeline_runs/figures")
    parser.add_argument("--prefix", type=str, default="yelp_soft_demo")
    parser.add_argument(
        "--hparam_group",
        type=str,
        default="hparam_transition_strength",
        choices=list(HP_PATTERNS.keys()),
    )
    parser.add_argument("--skip_ablation", type=int, default=0, choices=[0, 1])
    parser.add_argument("--skip_hparam", type=int, default=0, choices=[0, 1])
    return parser.parse_args()


def main() -> None:
    _paper_style()
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = _read_csv(args.csv)
    saved: List[str] = []

    if args.skip_ablation == 0:
        if args.csv_b.strip():
            rows_b = _read_csv(args.csv_b)
            saved.extend(
                plot_ablation_metric_split_two_datasets(
                    rows,
                    rows_b,
                    args.dataset_a_label,
                    args.dataset_b_label,
                    args.out_dir,
                    args.prefix,
                )
            )
        else:
            saved.append(plot_ablation_grouped_by_metric(rows, args.out_dir, args.prefix))
    if args.skip_hparam == 0:
        saved.append(plot_single_hparam_dual_axis(rows, args.hparam_group, args.out_dir, args.prefix))

    print("Saved:")
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
