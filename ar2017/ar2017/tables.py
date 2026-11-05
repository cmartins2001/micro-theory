"""Tables 1 and 2 of A&R (2017) for any combination of data sources, windows and instruments.

Every IV column is reported twice:
  * the A&R specification (one instrument per birthrate window; overidentified), and
  * a just-identified version (instrument = mean of the same windows; columns *_ji).
With several instruments the AR set tests beta and instrument validity jointly, so when J is
near rejection it can exclude zero while the Wald interval does not. The just-identified AR set
is a pure weak-IV-robust test of beta and is the one to read significance from.
"""
import numpy as np
import pandas as pd

from .est import ols, iv, ar_ci
from .data import region_dummies

CTRL = ['init', 'lpop', 'base']


def _iv_cols(data, X, Z, cluster=None):
    b, se, n, F, pJ = iv(data, 'dy', 'aging', X, Z, cluster=cluster)
    lo, hi, st = ar_ci(data, 'dy', 'aging', X, Z, cluster=cluster)
    data = data.assign(zbar=data[Z].mean(axis=1)) if 'zbar' not in data else data
    b1, se1, _, F1, _ = iv(data, 'dy', 'aging', X, ['zbar'], cluster=cluster)
    lo1, hi1, st1 = ar_ci(data, 'dy', 'aging', X, ['zbar'], cluster=cluster)
    return dict(b=b, se=se, N=n, F=F, J_p=pJ, n_inst=len(Z), AR_lo=lo, AR_hi=hi, AR_status=st,
                b_ji=b1, se_ji=se1, F_ji=F1, AR_ji_lo=lo1, AR_ji_hi=hi1, AR_ji_status=st1)


def t1_frame(D, gsrc, dsrc, t0, tg, ta, var, zstarts):
    d = D.base_frame()
    d['init'] = D.lgdp(gsrc, t0)
    d['dy'] = D.lgdp(gsrc, tg) - d.init
    d['base'] = D.dep(dsrc, var, t0)
    d['aging'] = D.dep(dsrc, var, ta) - d.base
    d['lpop'] = D.lpop(gsrc, t0)
    Z = []
    for i, a in enumerate(zstarts):
        d[f'z{i}'] = D.br(dsrc, a)
        Z.append(f'z{i}')
    R = region_dummies(d.region)
    return pd.concat([d, R], axis=1), list(R.columns), Z


def t1(D, gsrc, dsrc, t0, tg, ta, var, zstarts, tag):
    """Table 1: cols 1-4, 6-7 OLS (HC1); 5 and 8 IV. tg = GDP end year, ta = aging end year."""
    d, rn, Z = t1_frame(D, gsrc, dsrc, t0, tg, ta, var, zstarts)
    o = d[d.oecd == 1]
    rows = []
    for col, data, X in [(1, d, []), (2, d, ['init']), (3, d, CTRL), (4, d, CTRL + rn), (6, o, []), (7, o, CTRL)]:
        b, se, n = ols(data, 'dy', ['aging'] + X)
        rows.append(dict(col=col, b=b['aging'], se=se['aging'], N=n))
    for col, data, X in [(5, d, CTRL + rn), (8, o, CTRL)]:
        rows.append(dict(col=col, **_iv_cols(data, X, Z)))
    return [dict(table=1, **tag, **r) for r in rows]


def t2_frame(D, gsrc, dsrc, tg2, ta2, z2starts):
    """Stacked 1965-90 and 1990-end. Instruments are period-specific (zA: 1950-65 births for
    period 1; zB: z2starts for period 2). Only initial GDP gets a period-specific slope
    (Stata c.lgdp##i.period) -- the one specification that reproduces the paper."""
    parts = []
    for per, (t0, tg, ta, zs, pre) in {1: (1965, 1990, 1990, [1950, 1955, 1960], 'zA'),
                                       2: (1990, tg2, ta2, z2starts, 'zB')}.items():
        d = D.base_frame().assign(p2=float(per == 2))
        d['lgdp'] = D.lgdp(gsrc, t0)
        d['dy'] = D.lgdp(gsrc, tg) - d.lgdp
        d['base'] = D.dep(dsrc, 'dependency_50', t0)
        d['aging'] = D.dep(dsrc, 'dependency_50', ta) - d.base
        d['lpop'] = D.lpop(gsrc, t0)
        own = pd.DataFrame({f'{pre}{i}': D.br(dsrc, a) for i, a in enumerate(zs)}, index=d.index)
        d['zbar'] = own.mean(axis=1)           # just-identified instrument: own-period mean
        parts.append(pd.concat([d, own], axis=1))
    S = pd.concat(parts, ignore_index=True).dropna(subset=['dy', 'aging', 'lgdp', 'lpop', 'base'])
    Z = [c for c in S if c.startswith(('zA', 'zB'))]
    S[Z] = S[Z].fillna(0.0)                    # an instrument is zero outside its own period
    S['lgdp_p2'] = S.lgdp * S.p2
    R = region_dummies(S.region)
    C = pd.get_dummies(S.cc, prefix='c', drop_first=True).astype(float)
    return pd.concat([S, R, C], axis=1), list(R.columns), list(C.columns), Z


def t2(D, gsrc, dsrc, tg2, ta2, z2starts, tag):
    S, rn, cn, Z = t2_frame(D, gsrc, dsrc, tg2, ta2, z2starts)
    base = ['p2', 'lgdp', 'lgdp_p2', 'lpop', 'base']
    O = S[S.oecd == 1]
    rows = []
    for col, data, X in [(1, S, ['p2']), (2, S, base + rn), (3, S, base + cn), (5, O, base + cn)]:
        b, se, n = ols(data, 'dy', ['aging'] + X, cluster='cc')
        rows.append(dict(col=col, b=b['aging'], se=se['aging'], N=n))
    for col, data in [(4, S), (6, O)]:
        rows.append(dict(col=col, **_iv_cols(data, base + cn, Z, cluster='cc')))
    return [dict(table=2, panel='', **tag, **r) for r in rows]


def all_tables(D, gsrc, dsrc, end_g, end_a, zstarts, tag):
    """Panels A-C of Table 1 plus Table 2 for one run."""
    rows = []
    rows += t1(D, gsrc, dsrc, 1990, end_g, end_a, 'dependency_50', zstarts, dict(panel='A', **tag))
    rows += t1(D, gsrc, dsrc, 1990, end_g, end_a, 'dependency_ave', zstarts, dict(panel='B', **tag))
    rows += t1(D, gsrc, dsrc, 2000, end_g, end_a, 'dependency_50', zstarts, dict(panel='C', **tag))
    rows += t2(D, gsrc, dsrc, end_g, end_a, zstarts, tag)
    return rows
