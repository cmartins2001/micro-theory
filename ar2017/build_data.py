"""Build inputs for extend.py from PWT 11.0 and UN WPP 2024 (Medium variant).

Revised from the previous session's build_data.py. Output files and columns used
downstream are unchanged (pwt11.pkl, pwt11_mod.pkl, wpp_age.pkl, wpp_cbr.pkl);
changes are additive:
  * POP_DATE selects the age-structure file: 'jan' (Population1January..., the old
    behaviour) or 'jul' (PopulationByAge5GroupSex..., July 1 = the UN's standard
    mid-year convention, which is what the original 5-year UN series use).
  * wpp_cbr5.pkl: 5-year birthrates for windows [t, t+5), both as the simple mean
    of annual CBR (old method) and population-weighted (= births / person-years,
    the UN's definition of a period CBR).
  * Integrity checks: unique (ISO3, year, age group) rows, 21 age groups per
    country-year, and all sample countries present in every source.

Usage: python build_data.py [--pop-date jan|jul]   (paths come from ar2017/config.py)
QC found jan vs jul and simple vs weighted CBR immaterial (corr. with originals changes < 0.001);
defaults keep the previous session's choices (jan, simple mean in data.Data.br).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

AGE_FILES = {'jan': 'WPP2024_Population1JanuaryByAge5GroupSex_Medium.csv.gz',
             'jul': 'WPP2024_PopulationByAge5GroupSex_Medium.csv.gz'}
IND_FILE = 'WPP2024_Demographic_Indicators_Medium.csv.gz'
LAST_YEAR = 2024
N_AGE_GROUPS = 21   # 0-4, 5-9, ..., 95-99, 100+


def find(raw, name):
    """WPP file names come with either '_csv.gz' or '.csv.gz'; accept both."""
    for cand in (name, name.replace('_csv.gz', '.csv.gz'), name.replace('.csv.gz', '_csv.gz')):
        if (raw / cand).exists():
            return raw / cand
    raise FileNotFoundError(f'{name} not found in {raw}')


def build_pwt(raw, out):
    pw = pd.read_excel(raw / 'pwt110.xlsx', sheet_name='Data',
                       usecols=['countrycode', 'year', 'rgdpna', 'pop', 'emp', 'rnna',
                                'ck', 'cn', 'cgdpo', 'labsh'])
    assert not pw.duplicated(['countrycode', 'year']).any()
    pw[['countrycode', 'year', 'rgdpna', 'pop', 'emp', 'rnna', 'ck']].to_pickle(out / 'pwt11.pkl')
    pw[['countrycode', 'year', 'emp', 'cn', 'rnna', 'labsh', 'cgdpo']].to_pickle(out / 'pwt11_mod.pkl')
    return set(pw.countrycode)


def age_measures(p):
    """p: ISO3_code, Time, AgeGrpStart, PopTotal (5-year groups, 100+ coded 100)."""
    key = ['ISO3_code', 'Time']
    assert not p.duplicated(key + ['AgeGrpStart']).any(), 'duplicate age-group rows'
    ng = p.groupby(key).AgeGrpStart.nunique()
    assert (ng == N_AGE_GROUPS).all(), f'incomplete age groups: {ng[ng != N_AGE_GROUPS].head()}'
    old = p[p.AgeGrpStart >= 50].groupby(key).PopTotal.sum()
    yng = p[p.AgeGrpStart.between(20, 45)].groupby(key).PopTotal.sum()
    a20 = p[p.AgeGrpStart >= 20].assign(mid=lambda x: x.AgeGrpStart + 2.5)  # 100+ -> 102.5
    ave = (a20.mid * a20.PopTotal).groupby([a20.ISO3_code, a20.Time]).sum() / \
        a20.groupby(key).PopTotal.sum()
    return pd.DataFrame({'dep50': old / yng, 'ave20': ave}).reset_index()


def cbr_windows(d, starts=range(1950, 2020, 5)):
    """5-year birthrate over [t, t+5): simple mean and person-year weighted."""
    rows = []
    for t in starts:
        w = d[(d.Time >= t) & (d.Time < t + 5)]
        g = w.groupby('ISO3_code')
        n = g.Time.nunique()
        wtd = (w.CBR * w.TPopulation1July).groupby(w.ISO3_code).sum() / g.TPopulation1July.sum()
        rows.append(pd.DataFrame({'start': t, 'cbr_mean': g.CBR.mean(), 'cbr_wtd': wtd,
                                  'n_years': n}).reset_index())
    out = pd.concat(rows, ignore_index=True)
    return out[out.n_years == 5].drop(columns='n_years')


def main(raw, out, pop_date, sample=None):
    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pwt_codes = build_pwt(raw, out) if (raw / 'pwt110.xlsx').exists() else None

    p = pd.read_csv(find(raw, AGE_FILES[pop_date]),
                    usecols=['ISO3_code', 'Time', 'AgeGrpStart', 'PopTotal'], low_memory=False)
    p = p[p.ISO3_code.notna() & (p.Time <= LAST_YEAR)]
    age = age_measures(p)
    age.to_pickle(out / 'wpp_age.pkl')          # consumed by extend.py (same schema as before)
    age.to_pickle(out / f'wpp_age_{pop_date}.pkl')

    d = pd.read_csv(find(raw, IND_FILE),
                    usecols=['ISO3_code', 'Time', 'CBR', 'TPopulation1July'], low_memory=False)
    d = d[d.ISO3_code.notna() & (d.Time <= LAST_YEAR)]
    assert not d.duplicated(['ISO3_code', 'Time']).any()
    d.to_pickle(out / 'wpp_cbr.pkl')            # consumed by extend.py (same schema as before)
    cbr_windows(d).to_pickle(out / 'wpp_cbr5.pkl')

    if sample is not None:
        for name, codes in [('WPP age', set(age.ISO3_code)), ('WPP CBR', set(d.ISO3_code)),
                            ('PWT', pwt_codes)]:
            if codes is not None:
                miss = sorted(set(sample) - codes)
                assert not miss, f'{name} missing sample countries: {miss}'
    print(f'built ({pop_date}): {age.ISO3_code.nunique()} WPP locations, '
          f'years {age.Time.min()}-{age.Time.max()}')


if __name__ == '__main__':
    from ar2017 import config
    ap = argparse.ArgumentParser()
    ap.add_argument('--pop-date', choices=['jan', 'jul'], default='jan')
    a = ap.parse_args()
    main(config.RAW, config.BUILD, a.pop_date, list(pd.read_stata(config.raw_dta()).countrycode))
