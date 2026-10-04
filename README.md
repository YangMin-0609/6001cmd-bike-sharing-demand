# 6001CMD Bike Sharing Analysis

This repository supports the 6001CMD individual coursework using the UCI Bike Sharing Dataset (`hour.csv`). The primary machine-learning problem is hourly bike-demand regression, where `cnt` is the target variable.

## Dataset

- Source: [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- File used: `hour.csv`
- Licence shown by UCI: CC BY 4.0
- Observations: 17,379 hourly records
- Period: 1 January 2011 to 31 December 2012

The dataset file is intentionally excluded from Git. Download it from UCI and save it as `data/hour.csv`.

## Project structure

```text
6001CMD-bike-sharing/
|-- data/                         # Local dataset (not committed)
|-- figures/                      # Exported charts for the report
|-- notebooks/
|   `-- bike_sharing_analysis.ipynb
|-- report/                       # Local report drafts (not committed)
|-- src/
|   |-- analysis.py               # Reproducible data audit
|   |-- weather_heatmap.py        # Weather/target correlation figure
|   `-- report_figures.py         # Workflow and validation comparison figures
|-- tables/                       # Exported evidence tables
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## Setup in VS Code

1. Open this folder in VS Code.
2. Select `.venv\\Scripts\\python.exe` as the Python interpreter.
3. Open `notebooks/bike_sharing_analysis.ipynb` and select the same `.venv` kernel.
4. Run `src/analysis.py` or run the notebook cells from top to bottom.

## Key modelling safeguards

- Use a chronological train/validation/test split rather than a random split.
- Do not use `casual` or `registered` as predictors because `cnt = casual + registered`; including them would cause target leakage.
- Do not use `instant` as a predictor because it is only a row identifier.
- Report regression metrics such as MAE, RMSE, and R-squared; SMOTE, accuracy, precision, recall, and confusion matrices are not appropriate for the main regression task.

## Reproducibility

Create the environment from `requirements.txt`, place the UCI `hour.csv` file in `data/`, then run:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src\analysis.py
.venv\Scripts\python.exe src\weather_heatmap.py
.venv\Scripts\python.exe src\report_figures.py
```

The audit writes evidence tables to `tables/`. The additional scripts write report figures to `figures/`; the notebook produces the remaining exploratory charts and the Ridge validation output. Run its cells from top to bottom to reproduce those results. The validation chart uses the numerical results recorded in the notebook; it is a visualisation of that comparison, not an additional model experiment.
