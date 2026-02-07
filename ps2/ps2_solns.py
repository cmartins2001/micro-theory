import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    # from scipy import stats
    return mo, pd


@app.cell
def _(mo):
    mo.md(r"""
    ### Load Wage Data
    """)
    return


@app.cell
def _(pd):
    # Read wage CSV:
    df = pd.read_csv('wage.csv')
    return (df,)


@app.cell
def _(df):
    df
    return


if __name__ == "__main__":
    app.run()
