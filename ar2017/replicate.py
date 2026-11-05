"""Replicate A&R (2017) Tables 1-2 from the original data and diff every cell against the paper.
Output: results/replication_results.csv. Exits non-zero if any cell differs at the paper's precision."""
import sys
import warnings

import numpy as np
import pandas as pd

from ar2017 import config
from ar2017.data import Data, windows
from ar2017.tables import all_tables

warnings.filterwarnings('ignore')

# Paper values: (b, se) per column; (F, J p) for IV columns.
PAPER = {
    ('1', 'A'): [(0.335, .210), (1.036, .257), (1.162, .276), (.773, .322), (1.703, .411), (-.262, .352), (.042, .346), (1.186, .458)],
    ('1', 'B'): [(.042, .015), (.090, .017), (.092, .018), (.065, .024), (.103, .023), (.017, .046), (.001, .034), (.059, .035)],
    ('1', 'C'): [(.051, .210), (.950, .238), (1.028, .282), (.215, .259), (.809, .453), (-.151, .294), (.009, .285), (.373, .463)],
    ('2', ''): [(.305, .196), (.346, .274), (.775, .491), (2.168, .510), (.072, .322), (.361, .292)],
}
PAPER_IV = {('1', 'A', 5): (19.36, .51), ('1', 'A', 8): (7.38, .44), ('1', 'B', 5): (26.77, .63), ('1', 'B', 8): (8.01, .38),
            ('1', 'C', 5): (19.95, .62), ('1', 'C', 8): (8.06, .10), ('2', '', 4): (8.64, .44), ('2', '', 6): (4.69, .41)}
PAPER_N = {('1', 6): 35, ('1', 7): 35, ('1', 8): 35, ('2', 5): 63, ('2', 6): 63}


def main():
    D = Data()
    res = pd.DataFrame(all_tables(D, 'orig', 'orig', 2014, 2015, windows(0),
                                  dict(run='B0 original data', end=2014, instruments='original')))
    key = list(zip(res.table.astype(str), res.panel.fillna('').astype(str)))
    res['b_paper'] = [PAPER[k][c - 1][0] for k, c in zip(key, res.col)]
    res['se_paper'] = [PAPER[k][c - 1][1] for k, c in zip(key, res.col)]
    res['F_paper'] = [PAPER_IV.get(k + (c,), (np.nan, np.nan))[0] for k, c in zip(key, res.col)]
    res['J_p_paper'] = [PAPER_IV.get(k + (c,), (np.nan, np.nan))[1] for k, c in zip(key, res.col)]
    res['N_paper'] = [PAPER_N.get((k[0], c), 169 if k[0] == '1' else 282) for k, c in zip(key, res.col)]
    ok = ((res.b.round(3) == res.b_paper) & (res.se.round(3) == res.se_paper) & (res.N == res.N_paper)
          & (res.F_paper.isna() | (res.F.round(2) == res.F_paper))
          & (res.J_p_paper.isna() | (res.J_p.round(2) == res.J_p_paper)))
    res['match'] = ok
    config.RESULTS.mkdir(exist_ok=True)
    res.to_csv(config.RESULTS / 'replication_results.csv', index=False)
    print(f'replication: {ok.sum()}/{len(ok)} columns match the paper')
    if not ok.all():
        print(res.loc[~ok, ['table', 'panel', 'col', 'b', 'b_paper', 'se', 'se_paper', 'F', 'F_paper']])
        sys.exit(1)


if __name__ == '__main__':
    main()
