# Third-party software

This repository contains no vendored third-party source code. All code under `pipeline/`, `scripts/` and `src/` was
written for this project. Methods taken from the literature are reimplemented, not copied:

- periodic numeric embeddings ("PLR"): Gorishniy, Rubachev, Babenko, *On Embeddings for Numerical Features in
  Tabular Deep Learning*, NeurIPS 2022 (`scripts/legs/nnx/models.py`);
- ResNet-style MLP for tabular data: Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data*,
  NeurIPS 2021 (`scripts/legs/nnx/models.py`).

The dependencies below are installed by `uv sync` from `pyproject.toml` / `uv.lock`; they are not redistributed.
Every licence listed is compatible with distributing this project under the GNU GPL v3.

| Package | Version | Licence (package metadata) | Role |
| --- | --- | --- | --- |
| catboost | 1.2.10 | Apache-2.0 | gradient-boosted legs |
| lightgbm | 4.7.0 | MIT | gradient-boosted legs |
| torch | 2.14.0 | BSD-3-Clause (bundled components under Apache-2.0 and others) | neural legs |
| polars | 1.44.2 | MIT | data processing |
| pyarrow | 25.0.1 | Apache-2.0 | parquet I/O |
| numpy | 2.5.3 | BSD-3-Clause (bundled parts 0BSD, MIT, Zlib, CC0-1.0) | numerics |
| pandas | 2.3.3 | BSD-3-Clause | data processing |
| scikit-learn | 1.9.0 | BSD-3-Clause | preprocessing, calibration |
| statsmodels | 0.15.0 | BSD-3-Clause | diagnostics |
| duckdb | 1.5.5 | MIT | data processing |
| boto3 | 1.43.75 | Apache-2.0 | challenge data download |
| minio | 7.2.20 | Apache-2.0 | challenge data download |
| s3fs | 2026.7.0 | BSD-3-Clause | challenge data download |
| holidays | 0.104 | MIT | calendar features |
| optuna | 5.0.0 | MIT | hyper-parameter search |
| xgboost | 3.4.1 | Apache-2.0 | development experiments |
| pytabkit | 1.7.3 | Apache-2.0 | development experiments |
| skorch | 1.4.0 | BSD-3-Clause | development experiments |
| tabpfn-client | 0.5.3 | Apache-2.0 | development experiments |
| joblib | 1.6.0 | BSD-3-Clause | model files |
| pyyaml | 6.0.3 | MIT | configuration |
| tqdm | 4.70.0 | MPL-2.0 AND MIT | progress bars |
| matplotlib | 3.11.2 | Matplotlib licence (PSF-style) | plots |
| seaborn | 0.13.2 | BSD-3-Clause | plots |
| plotly | 7.0.0 | MIT | plots |
| jupyterlab, ipykernel, ipywidgets | 4.6.3, 7.3.0, 8.1.9 | BSD-3-Clause | notebooks |
| pytest (dev) | 9.1.1 | MIT | tests |

Transitive dependencies are pinned in `uv.lock`; their licences can be listed with
`uv run python -c "import importlib.metadata as m; [print(d.name, d.metadata.get('License-Expression') or d.metadata.get('License')) for d in m.distributions()]"`.
