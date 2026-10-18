# Finding solar panels from electricity meter readings

**AIL × Neural Wave 2026 — Team SilCrst**

Can we tell whether a customer has solar panels using only their electricity meter readings?
This project uses the energy drawn from the grid every 15 minutes to estimate:

- whether a photovoltaic (PV) system is present;
- when it started producing electricity;
- its initial capacity in **kWp** (kilowatts peak, the system's rated power under standard conditions);
- whether it was expanded later, and its total capacity after each detected expansion.

The key distinction is that **grid withdrawal is not the building's total electricity use**.
Solar panels can supply part of that use directly, reducing the energy drawn from the grid.
We look for this reduction and how it changes over time. No external weather, solar irradiance
or customer location data is used, and no LLM runs in the inference pipeline.

The submitted model combines **rules based on solar-production patterns** with a **Vision
Transformer (ViT)** for estimating initial capacity. The rules decide whether PV is present,
when production started and whether the system was expanded. The network estimates its size,
with a separate rule retained for small systems that only become clearly visible after expansion.

Report: [project report](../report/report.pdf).

## Team

- Gianluca Viviano
- Andrea Migliore
- Etienne Orio
- Ferdinando Giordano
- Giovanni Elisei

## Run the submitted model

The trained model is included in `models/`: **no training is needed to generate predictions**.
Inference runs on CPU and requires no internet connection once the dependencies are installed.

Use Python 3.11–3.13 (tested with 3.13.7). Install the versions specified in `requirements.txt`:
the CatBoost capacity models use Python's pickle format and depend on compatible library versions.
The Transformer weights are stored separately in `models/vit_capacity.npz`. Its inference code
uses NumPy and SciPy, so **PyTorch and a GPU are not required** to run the submission.

Run these commands from the directory containing this README and `inference.py`
(the activation command below is for macOS/Linux):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python inference.py path/to/test_set.csv
```

Replace `path/to/test_set.csv` with the actual input file. Each row represents one 15-minute
reading, with these columns:

| Column | Meaning |
|---|---|
| `customer_id` | Customer identifier, read as text |
| `energy_consumption_kwh` | Energy drawn from the grid during the interval, in kWh |
| `timestamp_epoch` | Unix timestamp of the reading; seconds and milliseconds are supported |

The reader detects comma or semicolon separators. For semicolon-separated input, it also
supports comma decimal marks.

### Output files

The command writes two files to the **current working directory**, replacing files with the same
names if they already exist.

**`predictions.csv`** contains one row per customer, with these columns in this order:

| Column | Meaning |
|---|---|
| `customer_id` | Customer identifier from the input |
| `pv_installed` | `TRUE` if PV is detected, otherwise `FALSE` |
| `pv_installation_date` | Estimated date when the initial system started producing |
| `pv_capacity_kwp` | Estimated initial capacity, in kWp |

For customers without detected PV, the date and capacity fields are empty.

**`pv_extensions_predictions.csv`** contains one row per detected expansion:

| Column | Meaning |
|---|---|
| `customer_id` | Customer identifier from the input |
| `extension_date` | Estimated expansion date |
| `pv_capacity_after_extension_kwp` | Total capacity after the expansion, in kWp |

If no expansions are detected, this file contains only the header.

Both files use UTF-8, dates in `YYYY-MM-DD` format, decimal points and capacities rounded to two
decimal places. The separator is a **semicolon**, following the text of the project definition
(§4.1). Its examples and the challenge README use commas instead. To produce comma-separated
output, change `OUTPUT_SEP` in `pvdetect/output.py`.

## How the model works

| Task | Component used in the submission |
|---|---|
| Detect PV and estimate its start date | Standby-load and midday-reduction rules, with seasonal and weather-pattern checks |
| Estimate initial capacity | Vision Transformer, except for the small-system expansion case described below |
| Detect expansions | Changes in the duration of the midday reduction, plus a rule for initially weak systems |
| Estimate capacity after an expansion | Estimated growth applied to initial capacity; CatBoost for the initially weak-system case |

### 1. Put all readings on a consistent timeline

Each customer's readings are arranged into a table with one row per UTC day and 96 columns,
one for each 15-minute interval. UTC does not change with daylight saving time, so every day has
the same structure. This also prevents the clock change from shifting the solar pattern by an
hour, as it would in local time.

### 2. Look for a recurring midday reduction

At night, solar panels do not produce electricity. We use the customer's low nighttime readings
to estimate a baseline, called the **standby load**. On sunny days, PV can reduce midday grid
withdrawal well below this baseline, sometimes to zero.

For example, if a customer usually draws 1 kWh per 15-minute interval at night but repeatedly
draws almost nothing around midday, solar production is one possible explanation. A single dip
is not enough: changes in activity or unusual meter readings can produce similar patterns.

We call a day **PV-like** when at least 30% of the intervals between 09:00 and 14:00 UTC fall below
half the estimated standby load, provided the nighttime readings are not similarly low. The
main detection rule looks at the highest share of PV-like days in any 60-day window.

A second rule handles small PV systems in buildings with high electricity demand. Their panels
may reduce grid withdrawal without ever bringing it below standby. Here, we look for a sustained
drop in midday withdrawal relative to nighttime withdrawal. The drop must follow the daily shape
of solar production and appear on both weekdays and weekends.

### 3. Check whether the pattern is consistent with sunshine

PV customers exposed to similar weather tend to show stronger midday reductions on the same
days. We use this shared pattern to estimate daily sunshine conditions from the meter readings
themselves.

During development, customers with clear PV patterns in the unlabelled data were grouped by
how their daily signals varied together. This produced two groups with different sequences of
sunny and cloudy days. We treat them as two weather regions, without assigning geographic
locations. Each group's average signal, based on 66–94 customers, becomes a **daily sunniness
index**. These aggregate indices are included in `models/weather_index.npz`; the file does not
contain individual customers' consumption series.

Additional checks help reject misleading detections: frequent zero readings at night, midday
reductions that do not follow either region's sunshine pattern, and dips that shift by an hour
when local clocks change. The checks are combined with seasonal and solar-shape evidence:
human activity follows local schedules, while solar production does not shift with daylight
saving time.

### 4. Estimate when production started

We look for the point where a recurring PV pattern begins, accounting for the fact that it is
easier to observe in summer than in winter. A change-point model refines this initial estimate.
We also look back for weaker early signs of production, so a gradual or low-output start is not
mistaken for a later installation.

If the system appears to have been present before the readings began, we return the first day
of the series. Its actual earlier installation date cannot be recovered from these readings.

### 5. Estimate the initial capacity

Capacity is harder to infer than presence. Once solar production covers the building's demand,
grid withdrawal reaches zero. A larger PV system may still produce more electricity, but that
extra production is invisible in the withdrawal readings.

We estimate the initial capacity with a **Vision Transformer**, a neural network that compares
patterns across the customer's whole withdrawal history. Each day is represented by a compact
summary of its 96 readings and which readings are missing. Calendar information helps the
network interpret seasonal and weekly changes, while a separate input preserves the building's
absolute consumption level.

The network estimates how likely each day is to be before or after the start of production,
then compares the two periods to infer capacity. This internal estimate is used only to calculate
capacity; the installation date written to the output still comes from the rules in step 4.
Instead of producing just one possible size, the network predicts a distribution of plausible
capacities. We select the value that maximizes the expected official capacity score under that
distribution, within the capacity limits stored in `models/baseline.json`.

There are too few labelled PV customers to train a network on real examples. The network was
therefore trained in two stages, without using any of the 40 labelled development customers:

1. first to fill in hidden days of the unlabelled series, which teaches it what normal
   consumption looks like;
2. then to predict the known capacities of simulated PV systems added to 369 unlabelled customers
   classified as PV-free. Simulated solar production is subtracted from their actual readings,
   stopping at zero grid withdrawal.

The simulation uses a clear-day production curve and daily sunshine factors derived from the
consumption data. The curve is estimated from small PV systems whose production is fully used
on site, where a reduction in grid withdrawal can reveal the amount produced.

The exported weights are in `models/vit_capacity.npz`, and the inference implementation is in
`pvdetect/vit_capacity.py`. The network updates the initial capacity of customers already
identified as having PV. It does not change their PV verdict or reported installation date.

Five **CatBoost models** remain part of the pipeline. Each was trained on labelled customers and
its own set of 1,360 simulated systems; their predictions are combined using a geometric mean.
They supply the capacity after expansion for the initially weak systems described in step 6.
If `vit_capacity.npz` is missing, inference prints a warning and keeps the baseline capacity
estimates. Keep all supplied model files in place to run the submitted configuration.

### 6. Detect later expansions (bonus task)

An expansion can make the daily period of below-standby withdrawal longer. We look for a lasting
increase in this duration that cannot be explained by the season alone.

A separate rule handles small systems that only become obvious after an expansion. If we find
evidence of solar production before the initially detected start, that later event may be an
expansion of an existing system. This helps avoid reporting the expansion as the first installation.

For these initially weak systems, the initial capacity is estimated from the earlier sunny-day
reduction, and CatBoost estimates the total capacity after expansion. The Transformer does not
overwrite these estimates: on 147 simulated cases of this kind, it underestimated the expanded
system's size (capacity score 0.30, compared with 0.68 for CatBoost).

For ordinary expansions, the detector estimates a capacity increase relative to the preceding
system size. The new initial capacity from the Transformer therefore also affects these later
capacity estimates, even though the expansion-detection rules are unchanged. The submitted
configuration reports up to two strongest qualifying candidates across the input file, in
addition to events found by the initially weak-system rule.

## Results and limitations

We evaluated the model on the **40 labelled development customers**, of whom 22 have PV, using
**leave-one-customer-out validation**. Each customer is predicted in turn, with the main detection
threshold and baseline capacity models fitted using the other 39 customers' labels. The same
pretrained Transformer is used in every fold: it was not trained on these customers' readings
or labels. The predictions are then scored together using the official metric:

```text
Main score = F1 × (70 + 15 × D + 15 × P)
```

Here, **F1** measures how well the model detects PV without missing systems or raising false
alarms. **D** scores installation dates and **P** scores capacity estimates, both on correctly
detected PV systems. All three components range from 0 to 1; higher is better. The main score
has a maximum of 100, before the separate extension bonus.

| Main score | F1: detection | D: installation date | P: capacity | Median relative capacity error |
|---|---|---|---|---|
| **89.39** | 1.000 | 0.732 | 0.561 | 19% |

All 22 PV systems are detected, with no false alarms among the 18 customers without PV. The
median absolute date error is 1 day, and 16 of the 22 dates are within a week of the labelled date.
The median capacity error of 19% means that half the estimates are within 19% of the true capacity;
16 of the 22 are within 25%. With the CatBoost capacity models instead of the network, the same
detections and dates score 88.26 (P 0.486, median capacity error 24%).

The gain in capacity score is promising but uncertain: resampling the 22 PV customers gives an
interval of −0.06 to +0.22 for the change in P relative to CatBoost. This interval includes zero,
so the observed improvement does not establish that the Transformer will perform better on
every new set of customers.

We also checked how much scores vary with the customers being evaluated:

| Check | Result | Interpretation |
|---|---|---|
| Bootstrap resampling of the 40 customers | 95% interval: 83.0–93.9 | Variation when sampling customers with replacement from the development predictions |
| Repeated subsets of 20 development customers | 5th–95th percentile: 86.2–92.6 | Variation across smaller evaluation sets drawn from the same 40 customers |
| Extension evaluation on simulated test-like sets | 2.7–3.0 bonus points | Estimated bonus on synthetic scenarios, not a measured hidden-test result |

**These are development results, not hidden-test results.** The two score intervals are
optimistic because we made design choices while examining the same 40 customers. They do not
capture all the uncertainty about performance on new customers, and we expect a lower main
score on the hidden test. `scripts/report.py` reproduces the first two checks from the saved
leave-one-out predictions.

## Rebuild the baseline and evaluate the submitted model

The commands below rebuild the detection settings, production and weather models, and CatBoost
capacity models. Evaluation then also applies the **included Transformer weights** to score the
submitted configuration. These commands do not retrain the Transformer; its separate training
and export workflow is described afterwards. The organizers' data is not included in the repository.

Place the following files in a local data directory. If your copies use the later
`*_development.csv` names, rename them or create links using the names below:

```text
energy_consumption_dev.csv
pv_labels_dev.csv
pv_extensions_labels_dev.csv
energy_consumption_unlabelled_dev.csv
```

With the virtual environment activated, run the commands below from the same directory as
`inference.py`. Set `DATA_DIR` to your data directory. Intermediate files are written to `cache/`
and `outputs/`, both excluded from version control. The weather-building and baseline training
steps update their files in `models/`; `vit_capacity.npz` is kept as supplied.

```bash
DATA_DIR="/absolute/path/to/data"
python -m pip install -r requirements.txt pytest

# Read and cache the input files; report data-quality issues.
python scripts/prepare_data.py "$DATA_DIR"

# Group unlabelled customers into confident PV, confident non-PV and uncertain cases.
python scripts/pseudo_label.py "$DATA_DIR"

# Estimate solar production and shared daily sunshine patterns from the readings.
python scripts/build_production.py "$DATA_DIR"
python scripts/build_weather.py "$DATA_DIR"

# Generate five simulated training sets, one for each CatBoost capacity model.
for k in 0 1 2 3 4; do
    python scripts/augment.py "$DATA_DIR" --seed "$k" --out "augmented_w${k}.csv"
done

# Refit the baseline on all development customers (does not train the Transformer).
python scripts/train.py "$DATA_DIR" --augmented cache/augmented_w{0,1,2,3,4}.csv

# Evaluate the baseline and the submitted variant with Transformer capacities.
python scripts/evaluate.py "$DATA_DIR" --augmented cache/augmented_w{0,1,2,3,4}.csv
python scripts/report.py outputs/loo_details.csv

# Run unit and pipeline tests; these use synthetic data only.
python -m pytest tests
```

With `models/vit_capacity.npz` present, `scripts/evaluate.py` prints both the baseline and
Transformer results and saves the submitted variant's predictions to `outputs/loo_details.csv`.
`scripts/report.py` uses those saved predictions for the score summary and resampling checks.
The score printed by `scripts/train.py` is only an in-sample baseline check, not the final
validation score.

### Transformer training and export

The Transformer was trained with the team's separate PyTorch implementation (`pvdetect.vit`,
documented in that repository's `docs/vit.md`; a GPU is recommended for training). Those training
modules are not included in this submission directory. This directory contains the exported
weights, NumPy inference code and a conversion script.

If you have the team's `base_0.pt` checkpoint from the synthetic-training stage and PyTorch
installed, you can regenerate the inference weights with:

```bash
python scripts/export_vit_capacity.py path/to/base_0.pt --out models/vit_capacity.npz
```

This converts the checkpoint into NumPy arrays; it does not train or fine-tune the network.
Neither the original checkpoint nor PyTorch is needed to use the supplied `.npz` file.

Additional calibration and analysis tools are in `scripts/calibrate_*.py`, `scripts/eval_*.py`,
`scripts/feature_stats.py` and `scripts/robustness.py`. To try the development workflow without
AIL data, `scripts/make_synthetic.py` generates a fake dataset with the same schema.

## Repository structure

| Path | Content |
|---|---|
| `inference.py` | Entry point that reads the test data and writes the submission files |
| `pvdetect/` | Input parsing, feature extraction, PV detection, dates, capacity, expansions and output formatting |
| `pvdetect/vit_capacity.py` | Transformer inference and replacement of eligible initial capacity estimates |
| `models/baseline.json` | Detection settings, capacity limits, production-curve parameters and expansion settings |
| `models/vit_capacity.npz` | Exported Transformer weights for initial capacity |
| `models/baseline_capacity.pkl` | Five CatBoost capacity models retained for the small-system expansion rule and fallback |
| `models/weather_index.npz` | Aggregate daily sunniness indices |
| `devtools/` | Training utilities for simulated PV production and weather-pattern estimation |
| `scripts/` | Data preparation, training, evaluation, calibration and analysis commands |
| `scripts/export_vit_capacity.py` | Conversion of the team's PyTorch checkpoint into inference weights |
| `tests/` | Unit and pipeline tests using synthetic data |

## Data rules and AI usage

- The model uses only the provided consumption series and their timestamps. It uses no external
  weather, irradiance or location data, and makes no attempt to reidentify customers.
- No AIL source data is included in this repository. The files in `models/` contain trained
  parameters and aggregate signals.
- **No LLM is part of the inference pipeline.**
- We used agentic coding tools to write code, brainstorm the approach and draft the report.

## Credits

Open-source libraries: NumPy, pandas, SciPy, scikit-learn, CatBoost and pytest; PyTorch to train the
capacity network (not needed at inference).

During development, we also checked the model on the public HEAPO dataset (heat-pump households
in Zurich, CC BY 4.0) to assess how it behaved on a different type of customer. HEAPO is not used
by the submitted model and is not included in this repository.
