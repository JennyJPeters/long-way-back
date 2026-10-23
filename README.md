# long-way-back

Code for "It's a Long Way Back: Frictional Re-entry and the Child Penalty" (J. Peters).

## Data
IPUMS CPS (Basic Monthly and ASEC, 2015-2019)
Citation: Flood et al. (2025), IPUMS CPS Version 13.0, https://doi.org/10.18128/D030.V13.0.

## Software
Python 3.12.9 with the packages pinned in `requirements.txt` (`pip install -r requirements.txt`).
Run on a Mac with an Apple M3 Pro and 18 GB of memory.

## Run
```
cd code
python3 00_master.py                                       # model tables and figures, no raw data needed
python3 00_master.py --facts                               # also the data figures (needs the raw CPS)
python3 00_master.py --rebuild-data --facts --recalibrate  # everything from raw data
```
The default run uses the shipped moments and calibrated parameters in `data/clean/`.
Results go to `output/`. `expected_output/` holds the paper's tables and figures for comparison.

## Output map
| Exhibit | File in `output/` | Script |
|---|---|---|
| Calibration table | `tables/tab_calib_params.tex` | `17_make_model_tables.py` |
| Fit tables | `tables/tab_fit_targeted.tex`, `tab_fit_untargeted.tex` | `12_make_fit_tables.py` |
| Decomposition | `tables/tab_decomposition.tex` | `14_decompose.py` |
| GE decomposition | `tables/tab_decomposition_ge.tex` | `17_make_model_tables.py` |
| Counterfactuals | `tables/tab_counterfactuals.tex` | `17_make_model_tables.py` |
| Exit rates | `figures/exit_EN.png`, `exit_EU.png` | `05_exit_asymmetry.py` |
| Re-entry hazards | `figures/reentry_NU.png`, `reentry_UE.png` | `06_reentry_flows.py` |
| Entry asymmetry | `figures/entry_asymmetry.png` | `07_entry_asymmetry.py` |
| Child age profile | `figures/child_age_profile.png` | `08_child_age_profile.py` |
| Depreciation figure | `figures/dn_compounding.png` | `16_decompose_dn.py` |


## Seeds
Only the bootstrap (`04_bootstrap_moments.py`) is random; its seed is 12345.



