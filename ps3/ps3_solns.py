import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Advanced Applied Econometrics Problem Set 3

    *Connor Martins -- 2/16/2026*
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import statsmodels.api as stats
    return mo, pd, stats


@app.cell
def _(mo):
    mo.md(r"""
    # Question 1
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using OLS to estimate the effects of price on quantity demanded is an example of the inherent endogeneity of simultaneous systems of equations. This means that there are multiple equations that can accurately explain variations in the same economic outcome of interest. In this case, the outcome is equilibrium quantity and it can be accurately explained by _both_ the supply and demand curves. The idea is fairly simple: we know quantity supplied is equal to quantity demanded in equilibrium, so knowing quantity and price alone does not guarantee the ability to explain the effect of the latter on the former since the direction of this relationship varies in each of the accurate underlying models.

    Since both price and quantity are endogeneous across multiple equations here, introducing an exogeneous variable like a supply shock is necessary for causal inference. It helps us understand how equilibrium quantity changes, holding the demand curved fixed, as supply shifts outward or inward. Isolating this relationship reveals the shape of the demand curve and demonstrates the demand-specific effects of price on quantity in this model.

    If we are interested in inference and not just predictive accuracy, then using the model which generates the best sample and holdout forecast results is not necessarily the most robust approach. The OLS model by definition will simply minimize error, even if it harnesses the endogeneity of the system to do so. However, estimating the OLS model on the demand variables does not guarantee a consistent estimator because price--used as the "independent" variable in this setting--is still related to disturbances (like supply shocks) due to the existence of other accurate models. In other words, supply shocks will be a part of the error term of the OLS model but are also related to price itself, violating the zero conditional mean assumption.

    The 2SLS approach, even if less accurate, will account for the effects that supply shocks like production costs have on price through the supply model. The intuition behind this is that higher production costs typically push supply down, raising price if demand remains constant. This relationship is captured by the first stage of the 2SLS procudure, which involves regressing price on the supply shock variable. From there, we use the predicted values of price based on production costs as our "independent" (despite that price is technically endogeneous to the system) variable in the second stage, which predicts quantity demanded. This simple procedure captures the dynamics of both of the underlying models in question, addressing endogeneity in a way that the basic OLS model does not.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Question 2:
    """)
    return


@app.cell
def _(pd):
    # Load RAW wage data:
    raw = pd.read_csv('CARD.raw', delim_whitespace=True, header=None)
    raw
    return (raw,)


@app.cell
def _(raw):
    # Parse and create a list of the column names:
    cols_raw = "id nearc2 nearc4 educ age fatheduc motheduc weight momdad14 sinmom14 step14 reg661 reg662 reg663 reg664 reg665 reg666 reg667 reg668 reg669 south66 black smsa south smsa66 wage enroll KWW IQ married libcrd14 exper lwage expersq"
    cols_fmtd = [_ for _ in cols_raw.split(" ")]

    # Set the formatted column names to be the column names of the df:
    raw.columns = cols_fmtd
    raw
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Part (a)

    *The below is the OLS regression output when using robust standard errors:*
    """)
    return


@app.cell
def _(raw, stats):
    # Create the df of exogeneous variables:
    exog = raw[[
        'educ', 'exper', 'expersq', 'black',
        'south', 'smsa', 'reg661', 'reg662', 
        'reg663', 'reg664', 'reg665', 'reg666',
        'reg667', 'reg668', 'smsa66'
    ]]

    # Regress log wage on the exogenous variables:
    lwage_ols_model = stats.OLS(
        endog=raw['lwage'],
        exog=stats.add_constant(exog),
    )

    # Estimate results with HC-robust and non-roust SEs:
    _non_robust = lwage_ols_model.fit()
    _robust = lwage_ols_model.fit(cov_type='HC0')

    # Display output:
    _robust.summary()
    return (exog,)


@app.cell
def _(mo):
    mo.md(r"""
    There is no difference between the statistical significance of any of the regression coefficients across the standard error techniques. Using heteroskedasticity-robust standard errors did not lower the p-value of any coefficient such that the null hypothesis could no longer be rejected. The only two variables with insignificant coeffcients were the region dummies for regions 2-6 and the `smsa66` variable, which were also insignificant without using robust standard errors.

    If there were changes to the statistical significance of any of these coefficients, it would be wise to use the robust standard errors in order to account for endogeneity. In this case, choosing between techniques is largely arbitrary.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Part (b)
    """)
    return


@app.cell
def _(exog, raw, stats):
    # Prepare a new exogeneous df without education and with nearc4 (proximity to 4-yr college):
    exog_reduced = exog.drop(columns=['educ']).assign(nearc4=raw['nearc4'])

    # Regress education on the rest of the exogeneous variables from the previous model:
    educ_reduced_model = stats.OLS(
        endog=raw['educ'],
        exog=stats.add_constant(exog_reduced),
    )

    # Fit with and without robust standard errors:
    _non_robust = educ_reduced_model.fit()
    _robust = educ_reduced_model.fit(cov_type='HC0')
    non_robust_nearc4_beta = _non_robust.params.loc['nearc4']
    robust_nearc4_beta = _robust.params.loc['nearc4']

    # Display regression output:
    _robust.summary()
    return non_robust_nearc4_beta, robust_nearc4_beta


@app.cell(hide_code=True)
def _(mo, non_robust_nearc4_beta, robust_nearc4_beta):
    mo.md(f"""
    When regressing years of education on the rest of the exogeneous variables from (a) and the `nearc4` variable, there is evidence of a statistically significant relationship between a person's proximity to a 4-year school and the years of education they achieved. The coefficient without robust standard errors was {round(non_robust_nearc4_beta, 4)} and with robust standard errors it was {round(robust_nearc4_beta, 4)}--identical results. Both coefficients were statistically significant at all levels with a p-value of 0.000, suggesting no difference in inference across standard error calculation type.
    """)
    return


if __name__ == "__main__":
    app.run()
