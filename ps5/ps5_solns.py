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
    from sklearn.metrics import root_mean_squared_error
    import pandas as pd
    return mo, pd, root_mean_squared_error, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 1:

    Part (a)
    """)
    return


@app.cell
def _():
    first_diff = r"$\boldsymbol{E[\boldsymbol{Y_{1,1}}|\boldsymbol{D_i} = 1]} - \boldsymbol{E[\boldsymbol{Y_{1,0}}|\boldsymbol{D_i} = 1]}$"
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Part (b)
    """)
    return


@app.cell
def _():
    # Latex for part b:
    beta_hat_symbol = r"$\hat{\boldsymbol{\beta}}$"
    return (beta_hat_symbol,)


@app.cell
def _(beta_hat_symbol, mo):
    mo.md(f"""
    {beta_hat_symbol} will be equal to the same DiD estimator we calculated in (a), so -.19, since this is merely a linear regression setup with a binary treatment period variable and a binary treatment variable. Doing a linear regression on this 2x2 setup simply recovers the same difference in differences we conducted before.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Part (c)
    """)
    return


@app.cell
def _():
    # Latex for part b:
    gamma_hat_symbol = r"$\hat{\boldsymbol{\gamma}}$"
    return (gamma_hat_symbol,)


@app.cell(hide_code=True)
def _(gamma_hat_symbol, mo):
    mo.md(f"""
    The value of {gamma_hat_symbol} will measure the effect of being in the treated state (i.e., Wisconsin) on a county's average wage holding constant treatment period, which is effectively measuring the baseline differences between states regardless of treatment status. This implies comparing average wages in the pre-treatment period across states, yielding {gamma_hat_symbol} = 15.23 - 16.42 = -1.19.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 2:
    """)
    return


@app.cell
def _(pd):
    # Load RAW labor force dataset:
    raw = pd.read_csv('MROZ.raw', delim_whitespace=True, header=None)
    # raw

    # Parse and create a list of the column names:
    cols_raw = ("inlf   hours   kidslt6   kidsge6   age   educ   wage   repwage   " +

    "hushrs   husage   huseduc   huswage   faminc   mtr   motheduc   " +

    "fatheduc   unem   city   exper   nwifeinc   lwage   expersq"
    )
    cols_fmtd = [_ for _ in cols_raw.split("   ")]

    # Set the formatted column names to be the column names of the df:
    raw.columns = cols_fmtd
    raw['counter'] = range(len(raw))
    # raw

    # Create the holdout dataset:
    holdout_init = raw.sample(253).set_index('counter')

    # Assign the rest to the train dataframe:
    train = (
        raw
        .drop(holdout_init.index)
        .reset_index()
        .drop(columns=['counter'])
        .drop(columns=['index'])
    )

    # Drop the index from the holdout set:
    holdout = holdout_init.reset_index().drop(columns=['counter'])
    return holdout, train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Part (a)
    """)
    return


@app.cell
def _(holdout, root_mean_squared_error, stats, train):
    ### Regress inlf on controls to get the conditional expectation:
    exog_cols = [
        'kidslt6', 'kidsge6', 'age', 'educ', 
        'repwage', 'faminc', 'exper' 
    ]
    ols_model = stats.OLS(
        endog=train['inlf'],
        exog=stats.add_constant(train[exog_cols]),
    ).fit()

    # Predict inlf using the holdout set:
    holdout_pred = ols_model.predict(
        exog=stats.add_constant(holdout[exog_cols])
    )

    # RMSE over holdout:
    rmse = root_mean_squared_error(
        holdout['inlf'],
        holdout_pred
    )

    return exog_cols, rmse


@app.cell(hide_code=True)
def _(mo, rmse):
    mo.md(f"""
    The RMSE from the OLS predictions relative to the holdout actuals of `inlf` was {round(rmse, 4)}.
    """)
    return


@app.cell
def _():
    # Libraries for LASSO model:
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    return LassoCV, StandardScaler


@app.cell
def _(
    LassoCV,
    StandardScaler,
    exog_cols,
    holdout,
    root_mean_squared_error,
    train,
):
    ### Get predictions using LASSO:

    # Standardize the controls in both datasets:
    scaler = StandardScaler()
    exog_train_norm = scaler.fit_transform(train[exog_cols])
    exog_holdout_norm = scaler.transform(holdout[exog_cols])

    # Define the inlf variables across datasets:
    y_train = train['inlf'].values
    y_holdout = holdout['inlf'].values

    # Use LassoCV library to find the optimal penalty:
    lasso_model = LassoCV(
        cv=5, 
        random_state=42
    ).fit(exog_train_norm, y_train)

    # Generate predictions on the holdout data from LASSO params:
    lasso_pred = lasso_model.predict(exog_holdout_norm)

    # Calc LASSO RMSE:
    lasso_rmse = root_mean_squared_error(y_holdout, lasso_pred)
    return lasso_model, lasso_rmse


@app.cell(hide_code=True)
def _(lasso_rmse, mo):
    mo.md(f"""
    The RMSE from the predictions generated by the LASSO model was {round(lasso_rmse, 4)}, identical to the RMSE we got from the OLS model. 
    """)
    return


@app.cell
def _(exog_cols, lasso_model):
    # Filter to the regressors that LASSO didn't drop:
    surviving_vars = [
        (col, coef) for col, coef in zip(exog_cols, lasso_model.coef_)
        if coef != 0
    ]
    print(f"Surviving variables: {surviving_vars}")
    return


@app.cell(hide_code=True)
def _(lasso_model, mo):
    mo.md(f"""
    All 7 of the regressors stayed in the LASSO model even after setting the optimal penalty value, which was {round(lasso_model.alpha_, 4)}. Such a low penalty value implies that the LASSO model is basically just converging to the same OLS estimation with controls, hence the identical predictions and RMSE value.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Question 3:
    """)
    return


@app.cell
def _():
    # ...
    return


if __name__ == "__main__":
    app.run()
