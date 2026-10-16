# Conformal Policy Selection 

Codebase for *Conformal Policy Selection: When to Deploy AI Decisions for
Targeted Intervention*.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
bash run_all.sh
```

Outputs (figures + CSVs) are written to `results/`.

`replicate_experiments.ipynb` reproduces the reproducible experiments (synthetic
doubly-robust study and LaLonde). 

The diabetes experiments need the
protected EHR data and are not included there.

| Paper item | Script |
|---|---|
| Fig. 1 (synthetic, correctly specified) | `experiments/doubly_robust.py` |
| Fig. 2 + Table 1 (four doubly-robust scenarios) | `experiments/doubly_robust.py` |
| Fig. 3 (diabetes safety gate) | `experiments/diabetes_safety_gate.py` |
| SI LaLonde | `experiments/lalonde.py` (`--c 4000`) |
| SI Sec. 4, Table/Fig. S3 (baseline comparison) | `experiments/baseline_comparison.py` |
| SI Sec. 5, Table/Fig. S4 (heterogeneous gate) | `experiments/diabetes_safety_gate_hetero.py` |

`ConfSelect/` holds the selection algorithms (`weighted_BH`, `weighted_CS`,
`conformal_select`, `BH`, `eval_sel`), adapted from the reference implementation
of weighted conformalized selection by Ying Jin and Emmanuel Candès
(https://github.com/ying531/conformal-selection).

## Data

`data/lalonde_data.csv` is included. The diabetes EHR dataset is proprietary.
To run the diabetes experiments with the data, set `DIABETES_DATA_PATH` (or pass
`--data_path`); `run_all.sh` skips them otherwise.
