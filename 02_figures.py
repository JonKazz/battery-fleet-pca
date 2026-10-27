"""Figures for the battery-fleet PCA report. Reads the result CSVs only."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

BLUE = "#1f4e79"
ORANGE = "#c45c26"
GRAY = "#6e6e6e"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def diurnal(rows: list[dict[str, str]]) -> None:
    hour = [int(row["hour_begin"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    ax.plot(hour, [float(row["actual"]) for row in rows], color="black", lw=2.0, label="Actual fleet")
    ax.plot(
        hour,
        [float(row["pca_centered_k1"]) for row in rows],
        color=BLUE,
        lw=1.8,
        label="PCA, mean subtracted, k = 1",
    )
    ax.plot(
        hour,
        [float(row["svd_raw_k1"]) for row in rows],
        color=BLUE,
        lw=1.2,
        ls="--",
        label="SVD of raw MW, k = 1",
    )
    ax.plot(
        hour,
        [float(row["pca_standardized_k1"]) for row in rows],
        color=ORANGE,
        lw=1.6,
        label="PCA, standardized, k = 1",
    )
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour beginning (Central)")
    ax.set_ylabel("Fleet net output (MW)")
    ax.set_title("Average day, 1-28 July 2026")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "diurnal_k1.pdf")
    fig.savefig(RESULTS / "diurnal_k1.png", dpi=140)
    plt.close(fig)


def error_curve(rows: list[dict[str, str]]) -> None:
    series = {
        "svd_raw": ("SVD of raw MW", BLUE, "-"),
        "pca_centered": ("PCA, mean subtracted", GRAY, "--"),
        "pca_standardized": ("PCA, standardized", ORANGE, "-"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    for key, (label, color, style) in series.items():
        chosen = [row for row in rows if row["method"] == key and int(row["k"]) <= 40]
        ax.plot(
            [int(row["k"]) for row in chosen],
            [float(row["rel_frobenius_mw"]) for row in chosen],
            color=color,
            ls=style,
            lw=1.8,
            label=label,
        )
    ax.set_xlim(1, 40)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Components kept, k")
    ax.set_ylabel("Relative Frobenius error in MW")
    ax.set_title("Reconstruction error of the battery-by-hour matrix")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "error_vs_k.pdf")
    fig.savefig(RESULTS / "error_vs_k.png", dpi=140)
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    diurnal(read_rows(RESULTS / "diurnal_fleet_mw.csv"))
    error_curve(read_rows(RESULTS / "error_by_k.csv"))


if __name__ == "__main__":
    main()
