"""Validate results/ against the paper and against the previous session's CSVs (reference/).

1. Replication: every Table 1-2 cell matches the paper (replicate.py's 'match' column).
2. Unchanged specs reproduce the old CSVs to 1e-8: all original-instrument extension rows, OLS
   columns of the shifted-instrument runs, all interaction rows, and old robustness columns.
3. Intended changes are printed side by side: shifted-instrument IV columns (post-1990 windows dropped).
4. Invariants: no instrument window ends after 1990; no AR set is disjoint; empty sets flagged.
"""
import sys

import numpy as np
import pandas as pd

from ar2017 import config

TOL = 1e-8
fails = []


def check(name, cond, detail=''):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


def close(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= TOL)))


R, REF = config.RESULTS, config.REFERENCE

# 1 --------------------------------------------------------------------------------------------
rep = pd.read_csv(R / 'replication_results.csv')
check('Tables 1-2 match the paper', rep.match.all(), f'{rep.match.sum()}/{len(rep)} columns')

# 2-3 ------------------------------------------------------------------------------------------
new = pd.read_csv(R / 'extension_results.csv')
old = pd.read_csv(REF / 'extension_results.csv')
for t in (new, old):
    t['panel'] = t.panel.fillna('').astype(str)
    t['inst'] = np.where(t.instruments.str.startswith('shifted'), 'shifted', 'original')
key = ['table', 'panel', 'run', 'inst', 'col']
m = new.merge(old, on=key, suffixes=('', '_old'), validate='one_to_one')
check('extension: same 240 cells', len(m) == len(old) == len(new), f'{len(m)} merged')
iv_cols = m.F_old.notna()
same = m[(m.inst == 'original') | ~iv_cols]
for c in ['b', 'se', 'N']:
    check(f'unchanged specs: {c}', close(same[c], same[c + '_old']), f'{len(same)} rows')
s_iv = same[same.F_old.notna()]
for c in ['F', 'J_p', 'AR_lo', 'AR_hi']:
    check(f'unchanged IV specs: {c}', close(s_iv[c], s_iv[c + '_old']), f'{len(s_iv)} rows')
check('AR status consistent with old (empty <-> NaN bounds)',
      ((s_iv.AR_status == 'empty') == s_iv.AR_lo_old.isna()).all())

chg = m[(m.inst == 'shifted') & iv_cols]
print(f'\nIntended changes: {len(chg)} shifted-instrument IV cells (post-1990 windows dropped)')
show = chg[['run', 'table', 'panel', 'col', 'b_old', 'b', 'se_old', 'se', 'F_old', 'F', 'AR_lo_old', 'AR_hi_old', 'AR_lo', 'AR_hi']]
print(show.round(3).to_string(index=False), '\n')

# interactions
ni, oi = pd.read_csv(R / 'interaction_results.csv'), pd.read_csv(REF / 'interaction_results.csv')
num = oi.select_dtypes('number').columns
check('interaction_results reproduce old CSV', ni.shape == oi.shape and close(ni[num], oi[num])
      and (ni[['window', 'moderator', 'sample', 'est']].values == oi[['window', 'moderator', 'sample', 'est']].values).all(),
      f'{len(oi)} rows')
nr, orr = pd.read_csv(R / 'interaction_robustness.csv'), pd.read_csv(REF / 'interaction_robustness.csv')
k = ['window', 'moderator', 'sample']
mr = orr.merge(nr, on=k, suffixes=('_old', ''), validate='one_to_one')
cols = ['N', 'b_int', 'se_int', 'b_int_income', 'se_int_income', 'loo_weakest_t']
check('interaction_robustness reproduces old rows', len(mr) == len(orr) and all(close(mr[c], mr[c + '_old']) for c in cols)
      and (mr.loo_weakest_drop.fillna('') == mr.loo_weakest_drop_old.fillna('')).all(), f'{len(mr)} of {len(nr)} rows')

# 4 --------------------------------------------------------------------------------------------
ends = new.windows.str.split().explode().str.split('-').str[1].astype(int)
check('no instrument window ends after 1990', ends.max() <= 1990, f'latest window end {ends.max()}')
st = pd.concat([new.AR_status, new.AR_ji_status]).dropna()
check('no disjoint AR sets', not (st == 'disjoint').any(), st.value_counts().to_dict().__repr__())

print(f'\n{len(fails)} failure(s)' if fails else '\nALL CHECKS PASSED')
sys.exit(1 if fails else 0)
