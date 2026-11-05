"""Extension of A&R (2017) Tables 1-2 with PWT 11.0 and UN WPP 2024.

Runs
  B0-B3  Bridge: original window (GDP 1990-2014, aging 1990-2015) and instruments; swap data vintages.
  E2019, E2023  GDP and aging both end in the same year, with
      'original'           A&R's windows 1960-65 ... 1985-90, and
      'shifted +s, pre-1990'  windows shifted forward by s = end - 2015 to keep the cohort-to-age-band
                              mapping, dropping any window that ends after 1990 (births after the start
                              of the long difference can respond to growth over it). This leaves
                              5 windows for 2019 (1964-89) and 4 for 2023 (1968-88).
Every IV column also carries a just-identified version (*_ji); see ar2017/tables.py.
Output: results/extension_results.csv
"""
import warnings

import pandas as pd

from ar2017 import config
from ar2017.data import Data, windows, shifted_windows
from ar2017.tables import all_tables

warnings.filterwarnings('ignore')


def main():
    D = Data()
    rows = []
    for g, dm, label in [('orig', 'orig', 'B0 original data'), ('pwt11', 'orig', 'B1 PWT 11 only'),
                         ('orig', 'wpp', 'B2 WPP 2024 only'), ('pwt11', 'wpp', 'B3 PWT 11 + WPP 2024')]:
        rows += all_tables(D, g, dm, 2014, 2015, windows(0), dict(run=label, end=2014, instruments='original'))
    for end in [2019, 2023]:
        s = end - 2015
        for inst, starts in [('original', windows(0)), (f'shifted +{s}y, pre-1990', shifted_windows(end))]:
            tag = dict(run=f'E{end}', end=end, instruments=inst, windows=' '.join(f'{a}-{a + 5}' for a in starts))
            rows += all_tables(D, 'pwt11', 'wpp', end, end, starts, tag)
    res = pd.DataFrame(rows)
    res['windows'] = res['windows'].fillna(' '.join(f'{a}-{a + 5}' for a in windows(0)))
    config.RESULTS.mkdir(exist_ok=True)
    res.to_csv(config.RESULTS / 'extension_results.csv', index=False)
    print(f'extension: {len(res)} rows')


if __name__ == '__main__':
    main()
