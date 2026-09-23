# Graph-Smoothed Prediction Intervals

Code for estimating prediction intervals with limited data, applied to traffic speeds and COVID-19 hospital admissions. The experiments compare Global, Local, and Graph Smoothing intervals.

## Data

Download `metr-la.h5` and `pems-bay.h5` from the links in [DCRNN's data instructions](https://github.com/liyaguang/DCRNN#data-preparation). Place both files directly in `data/`.

The six smaller input files are included in `data/`:

- Traffic graphs and sensor coordinates from [DCRNN](https://github.com/liyaguang/DCRNN): `adj_mx.pkl`, `adj_mx_bay.pkl`, `graph_sensor_locations.csv`, and `graph_sensor_locations_bay.csv`.
- Hospital admissions and geographic graph from [EpiPatch](https://github.com/Alistair-Turcan/EpiPatch): `HHShosp.csv` and `HHShosp_adj.csv`.

Please credit the original data providers and follow their data-use terms.

## Run

Use Python 3.11. From the repository root, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Settings are at the top of each runner. Set `FIGURES_ONLY = False` to run the full experiments.

```bash
python src/run_traffic_sensor.py
python src/run_hospital_admissions.py
```

Traffic trains STGCN from scratch. A CUDA-enabled PyTorch installation is recommended for faster training. The hospital admissions experiment runs on CPU.

## Results

Each runner creates `result/` automatically and saves PDF figures, CSV results, and TXT tables. Rerunning an application replaces its outputs. Training results can vary across hardware. The traffic interval example is selected using test scores for illustration.
