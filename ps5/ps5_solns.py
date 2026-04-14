import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Advanced Applied Econometrics Problem Set 5

    *Connor Martins -- 4/13/2026*
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import statsmodels.api as stats
    import pandas as pd
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 1:

    Part (a)
    """)
    return


@app.cell
def _():
    first_diff = r"$\boldsymbol{E[\boldsymbol{Y_{1,0}}|\boldsymbol{D_i} = 1]} - \boldsymbol{E[\boldsymbol{Y_{1,1}}|\boldsymbol{D_i} = 1]}$"
    second_diff = r"$\boldsymbol{E[\boldsymbol{Y_{0,1}}|\boldsymbol{D_i} = 1]} - \boldsymbol{E[\boldsymbol{Y_{0,0}}|\boldsymbol{D_i} = 1]}$"
    return first_diff, second_diff


@app.cell
def _():
    diff_in_diff = round((16.72 - 15.23) - (18.10 - 16.42), 2)
    return (diff_in_diff,)


@app.cell
def _(diff_in_diff, first_diff, mo, second_diff):
    mo.md(f"""
    The point estimate in this case is the simple Difference-in-Difference estimator for a 2x2 potential outcomes framework, defined as the first difference, {first_diff}, less the second difference, {second_diff}. This is simply comparing the changes in the average outcome (county-level minimum wage in the state of interest) in the treated and untreated states (Wisconsin and Minnesota respectively) in the pre and post-treatment periods. This difference works out to be (16.72 - 15.23) - (18.10 - 16.42) = {diff_in_diff}. This would suggest that the policy which limited collective bargaining power had a slight, negative effect on average wage among treated counties relative to untreated counties.
    """)
    return


if __name__ == "__main__":
    app.run()
