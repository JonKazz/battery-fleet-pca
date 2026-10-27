# Battery fleet PCA

Course project on dimension reduction for the ERCOT battery fleet.
The matrix is hourly metered output for 297 batteries, 1-28 July 2026.
Negative megawatts mean charging.

`01_fleet_pca.py` compares the section 10.6 PCA pipeline with a section 10.4 truncated SVD.
`02_figures.py` draws the two figures from the result tables.
`report.pdf` is the writeup, and `report.tex` is its source.

The result tables and figures are in `results/`.
`02_figures.py` rebuilds the figures from those CSVs.
The data pull inside `01_fleet_pca.py` reads ClickHouse through the LambdaERCOT app, which is not in this repo.
