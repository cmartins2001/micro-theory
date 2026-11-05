"""Paths. Override the raw-data folder with the AR2017_RAW environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get('AR2017_RAW', ROOT / 'raw'))
BUILD = ROOT / 'build'          # intermediate .pkl files from build_data.py
RESULTS = ROOT / 'results'      # CSVs consumed by the deck
REFERENCE = ROOT / 'reference'  # previous session's CSVs, for validate.py

# A&R's original file and Eggertsson et al.'s copy carry the same variables; accept either.
DTA_NAMES = ['aging_growth_data.dta', 'AGING_GROWTH.dta']


def raw_dta():
    for n in DTA_NAMES:
        if (RAW / n).exists():
            return RAW / n
    raise FileNotFoundError(f'none of {DTA_NAMES} in {RAW}')


def wpp_file(stem):
    """WPP files are distributed as '<stem>.csv.gz'; older copies used '<stem>_csv.gz'."""
    for suffix in ('.csv.gz', '_csv.gz'):
        if (RAW / (stem + suffix)).exists():
            return RAW / (stem + suffix)
    raise FileNotFoundError(f'{stem}(.csv.gz|_csv.gz) not in {RAW}')
