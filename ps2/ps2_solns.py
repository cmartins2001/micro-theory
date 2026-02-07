import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import altair as alt
    import seaborn as so
    # from scipy import stats
    return mo, pd, so


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


@app.cell
def _(mo):
    mo.md(r"""
    # Question 1:
    """)
    return


@app.cell
def _(df):
    # Wage summary stats:
    wage_mean = df['wage'].mean()
    wage_var = df['wage'].var()

    # Educ stats:
    educ_mode = df['schooling'].mode()[0]
    return educ_mode, wage_mean, wage_var


@app.cell(hide_code=True)
def _(educ_mode, mo, wage_mean, wage_var):
    mo.md(f"""
    ### Part (a):

    Wage sample mean: {round(wage_mean, 3)}; Wage sample variance: {round(wage_var, 3)}

    The most common number of years of schooling is {round(educ_mode, 3)}. These are presumably people who went through the traditional 12 years of education in the United States' primary school and high school and then went on to earn a 4-year bachelor's degree.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Part (b)
    """)
    return


@app.cell
def _(df, so):
    # Scatterplot of wage vs. schooling:
    so.scatterplot(x=df['schooling'], y=df['wage'])
    return


@app.cell
def _(mo):
    mo.md(r"""
    Based on the scatterplot, there seems to be some evidence of a positive, linear relationship between wage and schooling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 2:
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
