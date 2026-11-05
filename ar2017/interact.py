"""Theory test: does the aging-growth relationship strengthen where the model says it should?

Model condition for aging to raise output: theta > (1-theta)*eps_Gamma + 1/D.
  D (capital-labor gap)   -> 1990 log capital per worker (PWT cn/emp); capital-output ratio (cn/cgdpo)
  theta (automated share) -> labor share = (1-eta)(1-theta) in the model, so LOW labor share = HIGH theta
Predictions: aging x lnK/L > 0, aging x K/Y > 0, aging x labor share < 0.

Spec: Table 1 col 4 (OLS) / col 5 (IV) + aging x centered moderator + moderator main effect. IV adds
birthrate x moderator instruments (Sanderson-Windmeijer F reported). OECD is OLS only.
Moderators always come from PWT 11 (also in the 'original data' window, whose outcome is PWT 9).

Outputs: results/interaction_results.csv, results/interaction_robustness.csv
"""
import warnings

import numpy as np
import pandas as pd

from ar2017 import config
from ar2017.data import Data, windows, region_dummies
from ar2017.est import ols_v, iv_v

warnings.filterwarnings('ignore')
MODS = ['lnKL', 'KY', 'labsh']
SPECS = [('orig', 'orig', 2014, 2015, 'original data, 1990-2014'), ('pwt11', 'wpp', 2014, 2015, 'new data, 1990-2014'),
         ('pwt11', 'wpp', 2019, 2019, 'new data, 1990-2019'), ('pwt11', 'wpp', 2023, 2023, 'new data, 1990-2023')]


def moderators(D):
    M = D.pwt_mod
    ini = M[(M.year >= 1990) & (M.year <= 1994)]
    p90 = M[M.year == 1990].set_index('countrycode')
    MOD = pd.DataFrame({'lnKL': np.log(p90.cn / p90.emp), 'KY': p90.cn / p90.cgdpo,
                        'labsh': ini.groupby('countrycode').labsh.mean()})        # 1990-94 mean raises coverage
    sd = M[(M.year >= 1990) & (M.year <= 2019)].groupby('countrycode').labsh.std()
    MOD['labsh_flat'] = (sd.reindex(MOD.index) < 1e-6).fillna(False)              # constant series = likely imputed
    return MOD


def frame(D, MOD, gsrc, dsrc, end_g, end_a, zst):
    d = D.base_frame()
    d['init'] = D.lgdp(gsrc, 1990)
    d['dy'] = D.lgdp(gsrc, end_g) - d.init
    d['base'] = D.dep(dsrc, 'dependency_50', 1990)
    d['aging'] = D.dep(dsrc, 'dependency_50', end_a) - d.base
    d['lpop'] = D.lpop(gsrc, 1990)
    Z = []
    for i, a in enumerate(zst):
        d[f'z{i}'] = D.br(dsrc, a)
        Z.append(f'z{i}')
    for m in MOD.columns:
        d[m] = D.cc.map(MOD[m]).values
    d['labsh_flat'] = d.labsh_flat.fillna(False).astype(bool)
    R = region_dummies(d.region)
    return pd.concat([d, R], axis=1), list(R.columns), Z


def sample(d, m, which, need):
    """Complete cases for everything the regression uses, so centering = estimation-sample mean."""
    s = d[d[[m] + need].notna().all(axis=1)].copy()
    if which == 'OECD':
        s = s[s.oecd == 1]
    if which == 'measured labsh':
        s = s[~s.labsh_flat]
    s['mc'] = s[m] - s[m].mean()
    s['ic'] = s.init - s.init.mean()
    s['aXm'] = s.aging * s.mc
    s['aXi'] = s.aging * s.ic
    return s


def run(d, rn, Z, m, which, est, tag):
    need = ['dy', 'aging', 'init', 'lpop', 'base'] + (Z if est == 'IV' else [])
    s = sample(d, m, which, need)
    ctrl = ['init', 'lpop', 'base', 'mc'] + (rn if which != 'OECD' else [])
    if est == 'OLS':
        b, V, n = ols_v(s, 'dy', ['aging', 'aXm'] + ctrl)
        F = {}
    else:
        for z in Z:
            s[z + 'm'] = s[z] * s.mc
        b, V, n, F = iv_v(s, 'dy', ['aging', 'aXm'], ctrl, Z + [z + 'm' for z in Z])
    p10, p90 = np.percentile(s.mc, [10, 90])
    out = dict(**tag, moderator=m, sample=which, est=est, N=n,
               b_aging=b['aging'], se_aging=np.sqrt(V.loc['aging', 'aging']),
               b_int=b['aXm'], se_int=np.sqrt(V.loc['aXm', 'aXm']),
               F_aging=F.get('aging'), F_int=F.get('aXm'), sd_mod=s[m].std())
    idx = ['aging', 'aXm']
    for lab, v in [('p10', p10), ('p90', p90)]:
        g = np.array([1, v])
        out[f'eff_{lab}'] = b[idx] @ g
        out[f'se_{lab}'] = float(np.sqrt(g @ V.loc[idx, idx].values @ g))
    return out


def t_int(s, ctrl, with_income):
    X = ['aging', 'aXm'] + (['aXi'] if with_income else []) + ctrl
    b, V, _ = ols_v(s, 'dy', X)
    return b.aXm / np.sqrt(V.loc['aXm', 'aXm'])


def robustness(d, rn, m, which, tag):
    """Horse race against aging x initial income; OECD leave-one-out (weakest |t| on aging x moderator)."""
    s = sample(d, m, which, ['dy', 'aging', 'init', 'lpop', 'base'])
    ctrl = ['init', 'lpop', 'base', 'mc'] + (rn if which == 'All' else [])
    b, V, n = ols_v(s, 'dy', ['aging', 'aXm', 'aXi'] + ctrl)
    row = dict(**tag, moderator=m, sample=which, N=n, b_int=b.aXm, se_int=np.sqrt(V.loc['aXm', 'aXm']),
               b_int_income=b.aXi, se_int_income=np.sqrt(V.loc['aXi', 'aXi']))
    if which == 'OECD':
        for suffix, inc in [('', False), ('_with_income', True)]:
            loo = []
            for c in s.cc:
                t = sample(d[d.cc != c], m, which, ['dy', 'aging', 'init', 'lpop', 'base'])
                loo.append((t_int(t, ctrl, inc), c))
            w = min(loo, key=lambda x: abs(x[0]))
            row[f'loo_weakest_t{suffix}'], row[f'loo_weakest_drop{suffix}'] = w
    return row


def main():
    D = Data()
    MOD = moderators(D)
    rows, rob = [], []
    for g, dm, eg, ea, lab in SPECS:
        d, rn, Z = frame(D, MOD, g, dm, eg, ea, windows(0))
        tag = dict(window=lab)
        for m in MODS:
            plan = [('All', ['OLS', 'IV']), ('OECD', ['OLS'])] + ([('measured labsh', ['OLS', 'IV'])] if m == 'labsh' else [])
            for which, ests in plan:
                for est in ests:
                    rows.append(run(d, rn, Z, m, which, est, tag))
        for m in ['lnKL', 'labsh']:
            for which in ['All', 'OECD']:
                rob.append(robustness(d, rn, m, which, tag))
    config.RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(config.RESULTS / 'interaction_results.csv', index=False)
    pd.DataFrame(rob).to_csv(config.RESULTS / 'interaction_robustness.csv', index=False)
    print(f'interactions: {len(rows)} rows; robustness: {len(rob)} rows')


if __name__ == '__main__':
    main()
