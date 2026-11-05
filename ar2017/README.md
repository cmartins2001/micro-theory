# A&R (2017) replication and extension pipeline

```bash
./run_all.sh        # build -> replicate -> extend -> interact -> validate (about 35 s)
```

## Layout
| Path | Role |
|---|---|
| `ar2017/config.py` | Paths; set `AR2017_RAW` to point at a different raw-data folder |
| `ar2017/est.py` | OLS / 2SLS / AR / multi-endogenous IV with Stata `reg` / `ivreg2` conventions |
| `ar2017/data.py` | Variable access for the original data, PWT 11 and WPP 2024; instrument windows |
| `ar2017/tables.py` | Table 1 (Panels A–C) and Table 2 for any data, window and instruments |
| `build_data.py` | PWT 11 and WPP 2024 → `build/*.pkl`, with integrity checks |
| `replicate.py` | Original data → `results/replication_results.csv`; fails unless every cell matches the paper |
| `extend.py` | Bridge (B0–B3) and extension (E2019 / E2023) runs → `results/extension_results.csv` |
| `interact.py` | Theory test → `results/interaction_results.csv`, `results/interaction_robustness.csv` |
| `validate.py` | Checks against the paper and the previous session's CSVs (`reference/`) |
| `deck/` | marimo slide deck and its HTML/PDF backups; see `deck/README.md` |
| `raw/` | `AGING_GROWTH.dta` (or `aging_growth_data.dta`), `pwt110.xlsx`, the two WPP 2024 `.csv.gz` files |

## Changes from the previous session's pipeline
1. **Shifted instruments are capped at the pre-period.** Birthrate windows that end after 1990 are dropped, because births after the start of the long difference can respond to growth over it. This leaves 5 windows for 2019 (1964–89) and 4 for 2023 (1968–88). The `instruments` label now reads `shifted +Ny, pre-1990`, and a `windows` column lists the windows used.
2. **Every IV column also has a just-identified version** (`b_ji`, `se_ji`, `F_ji`, `AR_ji_*`), whose instrument is the mean of the same windows. In Table 2 it is the mean of each period's own windows. With several instruments, the AR test checks the coefficient and instrument validity together. The just-identified AR set tests the coefficient alone and is the one to read significance from.
3. **AR sets carry a status:** `bounded`, `unbounded` (the set reaches the ±10 grid edge), `disjoint` or `empty`. The old code returned `bounded=True` for empty sets. An empty multi-instrument set means the test rejects every coefficient value, i.e. the overidentifying restrictions fail.
4. **No `exec`.** Loaders are a module, and all paths come from `config.py`.
5. **Interactions** center on the complete-case estimation sample. The robustness file now covers all four windows (16 rows, previously 8), and the OECD leave-one-out is reported both without and with the aging × income interaction.

`validate.py` confirms that every unchanged specification reproduces the old CSVs to 1e-8. That covers 224 of 240 extension cells, all 44 interaction rows and the 8 old robustness rows. The 16 changed cells are the shifted-instrument IV columns, which it prints side by side.

## Headline IV results (Table 1 Panel A; Wald and AR sets, multi-instrument vs. just-identified)
| Run | col 5 b (se) | col 5 AR, just-id | col 8 OECD b (se) | col 8 b just-id (se) | col 8 AR, just-id |
|---|---|---|---|---|---|
| Original | 1.703 (0.411) | [0.76, 3.29] | 1.186 (0.458) | 1.395 (0.580) | [0.31, 3.35] |
| New data, original window | 1.975 (0.454) | [0.99, 3.08] | 1.071 (0.668) | 1.358 (0.714) | [−0.00, 3.58] |
| 2019, original IVs | 1.725 (0.573) | [0.28, 3.13] | 0.838 (0.594) | 1.481 (0.879) | [−0.14, 5.18] |
| 2019, shifted pre-1990 | 1.696 (0.579) | [0.29, 3.03] | 0.768 (0.648) | 1.382 (0.860) | [−0.26, 4.48] |
| 2023, original IVs | 1.515 (0.553) | [−0.10, 2.89] | 0.660 (0.485) | 1.329 (1.062) | [−1.13, 9.71] |
| 2023, shifted pre-1990 | 1.528 (0.597) | [−0.04, 2.70] | 0.470 (0.607) | 1.033 (0.976) | [−1.39, 4.50] |
