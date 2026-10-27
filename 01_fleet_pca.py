"""PCA and a plain SVD low-rank approximation of the battery fleet.

The matrix has one row per Central hour and one column per battery.
Each entry is the hour's mean telem_net_output in MW.
Negative MW is charging. Only telem_status ON and ONTEST are included.
An hour with no such row is 0 MW.

Section 10.6 of the book is the practical PCA pipeline: subtract the mean
of each battery, optionally divide by its standard deviation, then take the
leading eigenvectors of the covariance. Those eigenvectors are also the
right singular vectors of the centered (and possibly scaled) matrix.

Section 10.4 is the truncated SVD of whatever matrix you hand it. Applied
to the raw MW matrix, that is the best rank-k approximation in megawatts,
and it does not center or standardize first.

Reconstruction for the PCA pipeline undoes the scaling and adds the mean
back, so every method is scored in MW.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import db  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
START = "2026-07-01"
END = "2026-07-28"
MIN_HOURS = 24
MIN_HSL = 5.0
KS = list(range(1, 21)) + [30, 40, 60, 80]


def pull_hourly() -> list[tuple]:
    # Partition is toYYYYMM(market_date). The date filter limits the scan to
    # July even though ORDER BY starts at resource_name.
    frame = db.client().query(
        """
        SELECT
          resource_name,
          market_date,
          toHour(ts) AS hour_begin,
          avg(telem_net_output) AS net_mw,
          avg(hsl) AS hsl
        FROM pano.fact_esr_soc
        WHERE market_date >= {s:Date} AND market_date <= {e:Date}
          AND telem_status IN ('ON', 'ONTEST')
        GROUP BY resource_name, market_date, hour_begin
        ORDER BY resource_name, market_date, hour_begin
        LIMIT 500000
        SETTINGS max_execution_time = 180,
                 timeout_before_checking_execution_speed = 0,
                 max_rows_to_read = 30000000,
                 max_result_rows = 500000,
                 result_overflow_mode = 'throw'
        """,
        parameters={"s": START, "e": END},
    )
    return frame.result_rows


def build_matrix(rows: list[tuple]) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Return X (hours x batteries), names, HSL, and hour labels."""
    hours = sorted({(str(r[1]), int(r[2])) for r in rows})
    by_name: dict[str, list] = {}
    for name, day, hour, mw, hsl in rows:
        by_name.setdefault(name, []).append((str(day), int(hour), float(mw), float(hsl)))

    kept = []
    for name, samples in by_name.items():
        hsl_med = float(np.median([s[3] for s in samples if s[3] > 0] or [0.0]))
        if len(samples) >= MIN_HOURS and hsl_med >= MIN_HSL:
            kept.append((name, hsl_med, samples))
    kept.sort(key=lambda item: item[0])

    hour_index = {key: i for i, key in enumerate(hours)}
    labels = [f"{day} HE{hour + 1:02d}" for day, hour in hours]
    x = np.zeros((len(hours), len(kept)), dtype=np.float64)
    hsl = np.zeros(len(kept), dtype=np.float64)
    names = []
    for j, (name, hsl_med, samples) in enumerate(kept):
        names.append(name)
        hsl[j] = hsl_med
        for day, hour, mw, _ in samples:
            x[hour_index[(day, hour)], j] = mw
    return x, names, hsl, labels


def truncated_svd(matrix: np.ndarray, k: int) -> np.ndarray:
    """Best rank-k approximation of matrix in the Frobenius norm (section 10.4)."""
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return (u[:, :k] * s[:k]) @ vt[:k]


def prepare(x: np.ndarray, standardize: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean subtraction, then optional standardization. Both are column-wise."""
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    scale = np.ones((1, x.shape[1]), dtype=np.float64)
    if standardize:
        scale = centered.std(axis=0, ddof=1, keepdims=True)
        scale = np.where(scale < 1e-8, 1.0, scale)
        centered = centered / scale
    return centered, mean, scale


def pca_reconstruct(x: np.ndarray, k: int, standardize: bool) -> tuple[np.ndarray, np.ndarray]:
    """Section 10.6, scored back in MW.

    Steps: subtract each battery's mean, optionally divide by its standard
    deviation, form the covariance, keep the k leading eigenvectors, project,
    then undo the scaling and add the mean back.
    """
    centered, mean, scale = prepare(x, standardize)
    n_obs = centered.shape[0]
    covariance = (centered.T @ centered) / (n_obs - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    basis = eigenvectors[:, order[:k]]
    scores = centered @ basis
    low = scores @ basis.T
    return low * scale + mean, eigenvalues


def frobenius(a: np.ndarray, b: np.ndarray) -> float:
    err = a - b
    denom = np.linalg.norm(a)
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(err) / denom)


def fleet_rmse(actual: np.ndarray, recon: np.ndarray) -> float:
    """RMSE of the fleet-total MW, the quantity a desk would add up."""
    return float(np.sqrt(np.mean((actual.sum(axis=1) - recon.sum(axis=1)) ** 2)))


def unit_rmse(actual: np.ndarray, recon: np.ndarray) -> float:
    """RMSE per battery, then the average across batteries. Each unit counts once."""
    per = np.sqrt(np.mean((actual - recon) ** 2, axis=0))
    return float(per.mean())


def energy_captured(values: np.ndarray, k: int, singular: bool) -> float:
    """Share of energy in the first k components.

    Singular values contribute their square. Covariance eigenvalues already
    are that energy, up to the shared 1/(n-1) factor.
    """
    weights = values**2 if singular else np.maximum(values, 0.0)
    total = float(np.sum(weights))
    if total == 0:
        return 0.0
    return float(np.sum(weights[:k]) / total)


def fleet_r2(actual: np.ndarray, recon: np.ndarray) -> float:
    observed = actual.sum(axis=1)
    rebuilt = recon.sum(axis=1)
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    if ss_tot == 0:
        return 0.0
    ss_res = float(np.sum((observed - rebuilt) ** 2))
    return 1.0 - ss_res / ss_tot


def hour_of_day_profile(labels: list[str], series: np.ndarray) -> np.ndarray:
    buckets = [[] for _ in range(24)]
    for label, value in zip(labels, series):
        # Labels look like "2026-07-01 HE07". HE is hour-ending, clock hour is HE-1.
        he = int(label[-2:])
        buckets[he - 1].append(value)
    return np.array([float(np.mean(bucket)) for bucket in buckets])


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cache = RESULTS / "hourly_mw.npz"
    if cache.is_file():
        blob = np.load(cache, allow_pickle=True)
        x = blob["x"]
        names = blob["names"].tolist()
        hsl = blob["hsl"]
        labels = blob["labels"].tolist()
    else:
        rows = pull_hourly()
        x, names, hsl, labels = build_matrix(rows)
        np.savez(cache, x=x, names=np.array(names), hsl=hsl, labels=np.array(labels))

    n_hours, n_units = x.shape
    raw_s = np.linalg.svd(x, compute_uv=False)
    _, centered_s = pca_reconstruct(x, k=1, standardize=False)
    _, scaled_s = pca_reconstruct(x, k=1, standardize=True)

    methods = {
        "svd_raw": ("Section 10.4 SVD of the raw MW matrix", raw_s, True),
        "pca_centered": ("PCA, mean subtracted", centered_s, False),
        "pca_standardized": ("PCA, mean subtracted and standardized", scaled_s, False),
    }

    table = []
    for k in KS:
        if k >= min(n_hours, n_units):
            continue
        raw = truncated_svd(x, k)
        centered, _ = pca_reconstruct(x, k, standardize=False)
        standardized, _ = pca_reconstruct(x, k, standardize=True)
        recon = {
            "svd_raw": raw,
            "pca_centered": centered,
            "pca_standardized": standardized,
        }
        for key, (label, spectrum, singular) in methods.items():
            built = recon[key]
            table.append(
                [
                    key,
                    label,
                    k,
                    round(energy_captured(spectrum, k, singular), 6),
                    round(frobenius(x, built), 6),
                    round(fleet_rmse(x, built), 3),
                    round(fleet_r2(x, built), 6),
                    round(unit_rmse(x, built), 3),
                ]
            )

    write_csv(
        RESULTS / "error_by_k.csv",
        ["method", "description", "k", "energy_in_decomposed_space", "rel_frobenius_mw", "fleet_rmse_mw", "fleet_r2", "mean_unit_rmse_mw"],
        table,
    )

    # One diurnal picture: the actual fleet, and three reconstructions at a few ranks.
    fleet = x.sum(axis=1)
    profile_rows = []
    ranks_for_day = [1, 3, 10]
    profiles = {"actual": hour_of_day_profile(labels, fleet)}
    for k in ranks_for_day:
        profiles[f"svd_raw_k{k}"] = hour_of_day_profile(labels, truncated_svd(x, k).sum(axis=1))
        built, _ = pca_reconstruct(x, k, standardize=False)
        profiles[f"pca_centered_k{k}"] = hour_of_day_profile(labels, built.sum(axis=1))
        built, _ = pca_reconstruct(x, k, standardize=True)
        profiles[f"pca_standardized_k{k}"] = hour_of_day_profile(labels, built.sum(axis=1))
    for hour in range(24):
        profile_rows.append([hour, hour + 1] + [round(float(profiles[name][hour]), 3) for name in profiles])
    write_csv(
        RESULTS / "diurnal_fleet_mw.csv",
        ["hour_begin", "hour_ending"] + list(profiles),
        profile_rows,
    )

    mean_abs = np.abs(x.mean(axis=0))
    summary = RESULTS / "summary.csv"
    write_csv(
        summary,
        ["item", "value"],
        [
            ["window_start", START],
            ["window_end", END],
            ["n_hours", n_hours],
            ["n_batteries", n_units],
            ["mean_net_mw", round(float(x.mean()), 4)],
            ["median_abs_battery_mean_mw", round(float(np.median(mean_abs)), 4)],
            ["median_hsl_mw", round(float(np.median(hsl)), 3)],
            ["fleet_mean_mw", round(float(fleet.mean()), 3)],
            ["fleet_std_mw", round(float(fleet.std(ddof=1)), 3)],
        ],
    )
    print(f"hours {n_hours} batteries {n_units}")
    print(f"wrote {RESULTS / 'error_by_k.csv'}")


if __name__ == "__main__":
    main()
