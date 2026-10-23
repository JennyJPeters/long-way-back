"""Clean the IPUMS-CPS ASEC extract into the two frames used to estimate delta."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from paths import ASEC_CLEAN, ASEC_FILE

DAT = ASEC_FILE
OUT_DIR = ASEC_CLEAN

# Fixed-width columns, 0-based half-open (the IPUMS .do file is 1-based inclusive).
COLSPECS = {
    'year': (0, 4), 'serial': (4, 9), 'month': (9, 11), 'cpsid': (11, 25),
    'asecflag': (25, 26), 'asecwth': (26, 37), 'pernum': (37, 39),
    'cpsidp': (39, 53), 'cpsidv': (53, 68), 'asecwt': (68, 79),
    'age': (79, 81), 'sex': (81, 82), 'race': (82, 85), 'marst': (85, 86),
    'nchild': (86, 87), 'nchlt5': (87, 88), 'yngch': (88, 90), 'hispan': (90, 93),
    'empstat': (93, 95), 'labforce': (95, 96), 'educ': (96, 99),
    'classwly': (99, 101), 'workly': (101, 103), 'wkswork1': (103, 105),
    'uhrsworkly': (105, 108), 'wksunem1': (108, 110), 'fullpart': (110, 111),
    'incwage': (111, 119),
}

SKILL_EDUC_MIN = 111          # bachelor's or higher
PRIME_AGE = (25, 54)

# IPUMS NIU / missing sentinel codes.
INCWAGE_DROP_FROM = 99999998  # 99999998/99999999 = missing / NIU
YNGCH_NIU = 99                # 99 = no own child present
UHRS_VALID = (1, 99)          # usual hours/week last year; 999 = NIU

# CPI-U annual averages (BLS series CUUR0000SA0, 1982-84=100), used to deflate wages to 2018 dollars.
CPI_U = {2014: 236.736, 2015: 237.017, 2016: 240.007, 2017: 245.120, 2018: 251.107}
CPI_BASE = CPI_U[2018]


def load_raw() -> pd.DataFrame:
    if not DAT.exists():
        sys.exit(f"ASEC .dat not found at {DAT}")
    print(f"Reading {DAT.name} ...", flush=True)
    df = pd.read_fwf(DAT, colspecs=list(COLSPECS.values()),
                     names=list(COLSPECS.keys()))
    # Only the two ASEC weights carry implied decimals.
    df['asecwt'] = df['asecwt'] / 10000.0
    df['asecwth'] = df['asecwth'] / 10000.0
    print(f"  rows: {len(df):,}", flush=True)
    return df


def add_definitions(df: pd.DataFrame) -> pd.DataFrame:
    """Prime-age sample plus the group variables defined as in clean_cps.py."""
    df = df[df['age'].between(*PRIME_AGE) & df['sex'].isin([1, 2])].copy()
    df['is_female'] = (df['sex'] == 2).astype(int)          # 1 male, 2 female
    df['skill'] = (df['educ'] >= SKILL_EDUC_MIN).astype(int)
    df['has_child'] = (df['nchlt5'] >= 1).astype(int)
    df['group4'] = 2 * df['skill'] + df['is_female']        # 0=UnM 1=UnF 2=SkM 3=SkF
    df['mother'] = ((df['is_female'] == 1) & (df['has_child'] == 1)).astype(int)
    df['yngch_valid'] = df['yngch'].where(df['yngch'] != YNGCH_NIU)
    return df


def add_wage_and_states(df: pd.DataFrame) -> pd.DataFrame:
    """Add last-year real hourly wage and weeks employed / unemployed / out of the LF."""
    df = df.copy()
    df['earn_year'] = df['year'] - 1                        # ASEC asks about last year

    df['weeks_worked'] = df['wkswork1'].clip(lower=0, upper=52)
    # wksunem1 == 99 is NIU (full-year or non-workers), not 99 weeks; treated as 0.
    df['weeks_unemployed'] = (df['wksunem1'].where(df['wksunem1'] < 99, 0)
                              .clip(lower=0, upper=52))
    df['weeks_out_of_lf'] = (52 - df['weeks_worked'] - df['weeks_unemployed']).clip(lower=0)

    valid = (
        (df['incwage'] > 0) & (df['incwage'] < INCWAGE_DROP_FROM)
        & (df['weeks_worked'] > 0)
        & df['uhrsworkly'].between(*UHRS_VALID)
        & df['classwly'].between(20, 28)                   # wage and salary workers only
    )
    df['has_wage'] = valid.astype(int)

    hourly_nom = df['incwage'] / (df['weeks_worked'] * df['uhrsworkly'])
    cpi = df['earn_year'].map(CPI_U)
    df['hourly_wage_real'] = np.where(valid, hourly_nom * (CPI_BASE / cpi), np.nan)
    df['log_real_wage'] = np.log(df['hourly_wage_real'].where(df['hourly_wage_real'] > 0))
    return df


def build_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['year', 'earn_year', 'asecwt', 'log_real_wage', 'hourly_wage_real',
            'weeks_unemployed', 'weeks_out_of_lf', 'weeks_worked',
            'is_female', 'skill', 'has_child', 'group4', 'mother',
            'age', 'yngch_valid']
    out = df[(df['has_wage'] == 1) & df['log_real_wage'].notna()
             & (df['asecwt'] > 0)][cols].copy()
    return out.reset_index(drop=True)


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Pair each person's ASEC year a (wage1) with year a+1 (wage2) via cpsidp."""
    keep = df[df['cpsidp'] > 0].copy()
    base_cols = ['cpsidp', 'year', 'sex', 'age', 'asecwt', 'log_real_wage',
                 'has_wage', 'weeks_worked', 'weeks_unemployed', 'weeks_out_of_lf',
                 'skill', 'is_female', 'has_child', 'group4', 'mother']
    left = keep[base_cols].copy()
    right = keep[base_cols].copy()
    right['year_prev'] = right['year'] - 1

    merged = left.merge(
        right, left_on=['cpsidp', 'year'], right_on=['cpsidp', 'year_prev'],
        suffixes=('_1', '_2'))

    ok = ((merged['sex_1'] == merged['sex_2'])
          & (merged['age_2'] - merged['age_1']).between(0, 2)
          & (merged['has_wage_1'] == 1) & (merged['has_wage_2'] == 1)
          & merged['log_real_wage_1'].notna() & merged['log_real_wage_2'].notna())
    merged = merged[ok].copy()

    merged['dln_real_wage'] = merged['log_real_wage_2'] - merged['log_real_wage_1']
    out = pd.DataFrame({
        'cpsidp': merged['cpsidp'],
        'year2': merged['year_2'],
        'dln_real_wage': merged['dln_real_wage'],
        'weeks_employed': merged['weeks_worked_2'],
        'weeks_unemployed': merged['weeks_unemployed_2'],
        'weeks_out_of_lf': merged['weeks_out_of_lf_2'],
        'skill': merged['skill_2'], 'is_female': merged['is_female_2'],
        'has_child': merged['has_child_2'], 'group4': merged['group4_2'],
        'mother': merged['mother_2'], 'asecwt': merged['asecwt_2'],
    })

    # Winsorise the wage change at the 1st and 99th percentiles.
    lo, hi = out['dln_real_wage'].quantile([0.01, 0.99])
    out['dln_real_wage'] = out['dln_real_wage'].clip(lo, hi)
    return out.reset_index(drop=True)


def write_summary(cross: pd.DataFrame, panel: pd.DataFrame) -> None:
    lines = ["# ASEC cleaning summary\n",
             f"Source: data/raw/asec/cps_00013.dat (ASEC 2015-2019, earnings 2014-2018).",
             "Definitions identical to clean_cps.py (skill=educ>=111, is_female=sex==2,",
             "has_child=nchlt5>=1, prime-age 25-54). Wages: real 2018$ hourly, wage/salary.\n",
             "## Cross-section (delta_u, delta_n)\n",
             f"- n = {len(cross):,} prime-age wage/salary workers with a valid real wage",
             f"- share with any unemployment last year: "
             f"{(cross['weeks_unemployed']>0).mean():.3f}",
             f"- share with any out-of-LF weeks last year: "
             f"{(cross['weeks_out_of_lf']>0).mean():.3f}\n",
             "## Before/after panel (delta_e, delta_u, delta_n)\n",
             f"- n = {len(panel):,} linked consecutive-March person-pairs with a wage in both",
             f"- mean dln real wage: {panel['dln_real_wage'].mean():+.4f}",
             f"- by skill: non-college {panel[panel['skill']==0]['dln_real_wage'].mean():+.4f}, "
             f"college {panel[panel['skill']==1]['dln_real_wage'].mean():+.4f}\n",
             "NOTE: panel uses year-2 ASECWT (a dedicated ASEC longitudinal weight was",
             "not requested); linked sample is selected on being employed in both years",
             "(handle via indirect inference downstream).\n"]
    (OUT_DIR / 'asec_clean_summary.md').write_text("\n".join(lines))


def main() -> int:
    df = load_raw()
    df = add_definitions(df)
    df = add_wage_and_states(df)

    cross = build_cross_section(df)
    panel = build_panel(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cross.to_csv(OUT_DIR / 'asec_cross.csv', index=False)
    panel.to_csv(OUT_DIR / 'asec_panel.csv', index=False)
    write_summary(cross, panel)

    print(f"\nCross-section: {len(cross):,} workers  -> data/clean/asec/asec_cross.csv")
    print(f"Panel        : {len(panel):,} person-pairs -> data/clean/asec/asec_panel.csv")
    print(f"Summary      : data/asec/asec_clean_summary.md")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
