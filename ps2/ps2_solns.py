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
    import statsmodels.api as stats
    # from scipy import stats
    return mo, np, pd, so, stats


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
    ### Part (a)

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

    I don't exogeneity necessarily hold; we know that unobservable factors certainly affect wage, confounding the simple OLS model in question and introducing endogeneity. There is most likely an unobservable attribute of people in the dataset, such as an innate ability to make money, that is related to both X and Y, meaning that the error term of the OLS model will certainly be related

    Heteroskedasticity is likely an issue here. Without knowing more about the dataset, it's certainly possible that certain groups within the data have similar underlying characteristics that produce different variances in the independent variable, schooling, causing heteroskedasticity to be a problem.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Part (c)
    """)
    return


@app.cell
def _(df, np):
    # Calculate the sample covariance and correlations:
    sample_covar = np.cov(m=df['schooling'], y=df['wage'])[0,1]
    sample_corr = np.corrcoef(df['schooling'], df['wage'])[0,1]
    return sample_corr, sample_covar


@app.cell
def _(mo, sample_corr, sample_covar):
    mo.md(f"""
    The sample covariance between wage and schooling is {round(sample_covar, 3)}. The sample correlation between them is {round(sample_corr, 3)}.
    """)
    return


@app.cell
def _(df, stats):
    # Fit an OLS model:
    ols_model = stats.OLS(
        endog=df['wage'],
        exog=stats.add_constant(df['schooling'])
    )

    # Model Output:
    ols_beta = ols_model.fit().params.loc['schooling']
    ols_model.fit().summary()
    return (ols_beta,)


@app.cell
def _(mo, ols_beta):
    mo.md(f"""
    The slope estimate of this OLS model is {round(ols_beta, 5)}.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Part (d)
    """)
    return


@app.cell
def _(df, np):
    # Create a new wage variable:
    df['wage_10'] = df['wage'] * 10  # this is wage but in thousands of USD

    # Calculate sample correlation / covariance with wage_10:
    sample_covar_w10 = np.cov(m=df['schooling'], y=df['wage_10'])[0,1]
    sample_corr_w10 = np.corrcoef(df['schooling'], df['wage_10'])[0,1]
    return sample_corr_w10, sample_covar_w10


@app.cell
def _(mo, sample_corr_w10, sample_covar_w10):
    mo.md(f"""
    The sample covariance between `wage_10` and schooling is {round(sample_covar_w10, 3)}. The sample correlation between them is {round(sample_corr_w10, 3)}.
    """)
    return


@app.cell
def _(df, stats):
    # Run an OLS model on the wage_10 variable:
    ols_model_w10 = stats.OLS(
        endog=df['wage_10'],
        exog=stats.add_constant(df['schooling'])
    )

    # Model Output:
    ols_beta_w10 = ols_model_w10.fit().params.loc['schooling']
    ols_model_w10.fit().summary()
    return ols_beta_w10, ols_model_w10


@app.cell
def _(mo, ols_beta_w10):
    mo.md(f"""
    The slope estimate of the OLS model using `wage_10` is {round(ols_beta_w10, 5)}. We can see that using wage in thousands of USD had the effect of simply scaling the sample covariance, correlation, and OLS beta by 10. The new interpretation of the model is that a 1-year increase in schooling is associated with a $1,130 increase in wage.
    """)
    return


@app.cell
def _():
    # Latex code for derivation:
    ols_formula = r"$\hat{\boldsymbol{\beta_{OLS}}} = (\frac{1}{n}\Sigma\mathbf{X_i}\mathbf{X_i}')^{-1}(\frac{1}{n}\Sigma\mathbf{X_i}'\mathbf{y_i})$"
    y = r"$\mathbf{y_i}$"
    beta = r"$\hat{\boldsymbol{\beta_{OLS}}}$"
    ten_x_beta = r"$(\frac{1}{n}\Sigma\mathbf{X_i}\mathbf{X_i}')^{-1}(\frac{1}{n}\Sigma\mathbf{X_i}'\mathbf{(10*y_i))} = \hat{\boldsymbol{\beta_{OLS}}}*10$"
    return beta, ols_formula, ten_x_beta, y


@app.cell
def _(beta, mo, ols_formula, ten_x_beta, y):
    mo.md(f"""
    If The OLS estimator is {ols_formula}, then creating the `wage_10` variable was equivalent to multiplying {y} by 10. By the OLS formula for {beta}, we can show that {ten_x_beta}. This, in theory, matches our result above.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Question 2:
    """)
    return


@app.cell
def _():
    # ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Question 3:
    """)
    return


@app.cell
def _(np, ols_model_w10, pd, stats):
    # Prepare a custom function which return a list containing a column vector of OLS coefficients and the standard covariance estimation:

    def get_ols_and_covariance_estimates(X : pd.DataFrame, y : pd.Series):
    
        # Fit the OLS model:
        _ols_model = stats.OLS(
            endog=y,
            exog=stats.add_constant(X)
        ).fit()

        # Store OLS coefficients in a list:
        ols_coeffs_list = [_ for _ in ols_model_w10.fit().params]

        # Calculate the covariance estimate of the OLS parameters:
        cov_est = np.cov(np.array(ols_coeffs_list))

        return [ols_coeffs_list, cov_est]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Part (a)
    """)
    return


@app.cell
def _(mo):
    # Set vector length:
    N_input = mo.ui.number(value=5, start=1, stop=100000, step=1)
    N_input
    return (N_input,)


@app.cell
def _(N_input, np):
    # Monte carlo trials:
    N = N_input.value
    e = np.random.standard_normal(N)
    # e = np.array([-0.51316885, -1.80754049, -1.12065741,  0.45416927,  0.90478367]) # hard-coded for testing
    ones = np.array([1 for _ in range(N)])
    mat = np.array(np.array([ones, e])).T
    b = np.array([1,2]).T
    y_vector = (mat @ b) + e
    y_vector
    return


if __name__ == "__main__":
    app.run()
