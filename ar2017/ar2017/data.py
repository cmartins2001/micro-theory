"""Data access for the original A&R variables and the PWT 11 / WPP 2024 rebuilds.

Source codes used throughout:
  GDP:   'orig' (PWT 9.0 in the A&R file)  | 'pwt11'
  Aging: 'orig' (UN data in the A&R file)  | 'wpp'  (WPP 2024, built by build_data.py)
"""
from functools import cached_property

import numpy as np
import pandas as pd

from . import config

PRE_PERIOD_END = 1990   # instruments must be births that finish by the start of the long difference


def windows(shift=0, k=6, start=1960, cap=None):
    """Start years of the 5-year birthrate windows [a, a+5).

    shift=0 gives A&R's instruments (1960-65 ... 1985-90). Shifting forward by the change in end
    year keeps the mapping from birth cohorts to the 20-49 age band. cap drops windows that end
    after `cap`, so post-1990 births (which can respond to 1990+ growth) never enter the instrument set.
    """
    w = [start + shift + 5 * i for i in range(k)]
    return [a for a in w if cap is None or a + 5 <= cap]


def shifted_windows(end_year):
    """Extension instruments: shifted by (end_year - 2015) and capped at the pre-period."""
    return windows(end_year - 2015, cap=PRE_PERIOD_END)


class Data:
    def __init__(self, raw_dta=None, build_dir=None):
        self.raw_dta = raw_dta or config.raw_dta()
        self.build = build_dir or config.BUILD

    @cached_property
    def RAW(self):
        return pd.read_stata(self.raw_dta)

    @property
    def cc(self):
        return self.RAW.countrycode

    @cached_property
    def _pwt(self):
        p = pd.read_pickle(self.build / 'pwt11.pkl')
        return {v: p.pivot(index='countrycode', columns='year', values=v) for v in ['rgdpna', 'pop']}

    @cached_property
    def pwt_mod(self):
        return pd.read_pickle(self.build / 'pwt11_mod.pkl')

    @cached_property
    def _age(self):
        a = pd.read_pickle(self.build / 'wpp_age.pkl')
        return {'dependency_50': a.pivot(index='ISO3_code', columns='Time', values='dep50'),
                'dependency_ave': a.pivot(index='ISO3_code', columns='Time', values='ave20')}

    @cached_property
    def _cbr(self):
        return pd.read_pickle(self.build / 'wpp_cbr.pkl').pivot(index='ISO3_code', columns='Time', values='CBR')

    # ---- variables, all aligned to RAW row order ----
    def lgdp(self, src, y):
        if src == 'orig':
            return np.log(self.RAW[f'rgdpna{y}'] / self.RAW[f'pop{y}']).values
        return np.log(self.cc.map(self._pwt['rgdpna'][y]) / self.cc.map(self._pwt['pop'][y])).values

    def lpop(self, src, y):
        if src == 'orig':
            return np.log(self.RAW[f'pop{y}']).values
        return np.log(self.cc.map(self._pwt['pop'][y])).values

    def dep(self, src, var, y):
        """var: 'dependency_50' (pop 50+ / pop 20-49) or 'dependency_ave' (mean age of 20+)."""
        if src == 'orig':
            return self.RAW[f'{var}_y{y}'].values
        return self.cc.map(self._age[var][y]).values

    def br(self, src, a):
        """Birthrate over [a, a+5): original UN period rate, or mean of annual WPP CBR."""
        if src == 'orig':
            return self.RAW[f'birthrate{a}_{a+5}'].values
        return self.cc.map(self._cbr[list(range(a, a + 5))].mean(axis=1)).values

    def base_frame(self):
        return pd.DataFrame({'cc': self.cc, 'region': self.RAW.region, 'oecd': self.RAW.sample_oecd})


def region_dummies(region):
    return pd.get_dummies(region, prefix='r', drop_first=True).astype(float)
