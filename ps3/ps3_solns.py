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
    from linearmodels.iv import IV2SLS
    return IV2SLS, mo, np, pd, stats


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

    # Store confidence intervals for later:
    lwage_ols_beta_ci = list(lwage_ols_model.fit(cov_type='HC0').conf_int().loc['educ'])

    # Display output:
    _robust.summary()
    return exog, lwage_ols_beta_ci


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
    exog_reduced = stats.add_constant(exog.drop(columns=['educ']).assign(nearc4=raw['nearc4']))

    # Regress education on the rest of the exogeneous variables from the previous model:
    educ_reduced_model = stats.OLS(
        endog=raw['educ'],
        exog=exog_reduced,
    )

    # Fit with and without robust standard errors:
    _non_robust = educ_reduced_model.fit()
    _robust = educ_reduced_model.fit(cov_type='HC0')
    non_robust_nearc4_beta = _non_robust.params.loc['nearc4']
    robust_nearc4_beta = _robust.params.loc['nearc4']

    # Display regression output:
    _robust.summary()
    return (
        educ_reduced_model,
        exog_reduced,
        non_robust_nearc4_beta,
        robust_nearc4_beta,
    )


@app.cell(hide_code=True)
def _(mo, non_robust_nearc4_beta, robust_nearc4_beta):
    mo.md(f"""
    When regressing years of education on the rest of the exogeneous variables from (a) and the `nearc4` variable, there is evidence of a statistically significant relationship between a person's proximity to a 4-year school and the years of education they achieved. The coefficient without robust standard errors was {round(non_robust_nearc4_beta, 4)} and with robust standard errors it was {round(robust_nearc4_beta, 4)}--identical results. Both coefficients were statistically significant at all levels with a p-value of 0.000, suggesting no difference in inference across standard error calculation type.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Part (c)
    """)
    return


@app.cell
def _(educ_reduced_model, exog, exog_reduced, np, pd, raw, stats):
    ### To estimate the IV model, regress log(wage) on predicted values of education generated by reduced form on nearc4:

    # First, get predicted values of education from the first stage regression in (b):
    ''' educ = const + beta * x1 + beta * x2 + ... '''
    educ_pred = (
        educ_reduced_model
        .predict(
            exog=exog_reduced, # we regress on all covariates including nearc4, the IV
            params=educ_reduced_model.fit().params
        )
    )
    # pd.concat([raw['educ'], pd.Series(educ_pred, name='educ_pred')], axis=1) # view educ and educ_pred side by side

    # Construct the second stage design matrix -- drop the IV and add the predicted level of education:
    iv_exog = pd.concat([pd.Series(educ_pred, name='educ_pred'), exog_reduced.drop(columns=['const', 'nearc4'])], axis=1)

    # Then, regress log wage on predicted values of education from the first stage regression:
    ''' log(wage) = const + beta * educ_hat + beta * x1 + beta * x2 + ...'''
    nearc4_IV_ols = stats.OLS(
        endog=raw['lwage'],
        exog=stats.add_constant(iv_exog), # do not include the iv, nearc4, in the second stage
    )

    ### CORRECT VARIANCES OF THE IV ESTIMATORS SINCE THIS WAS A MANUAL 2SLS PROCEDURE:

    # Recalculate the residuals using the actual values of educ instead of educ_pred to avoid variance issues:
    X_actual = stats.add_constant(exog) # all the original exogeneous vars including educ
    IV_beta = nearc4_IV_ols.fit().params
    u_hat = raw['lwage'] - (X_actual @ np.array(IV_beta)) # the residuals from the IV model using actual, not predicted educ

    # Fix the variance-covariance matrix:
    n, k = X_actual.shape
    sigma_sq = (u_hat @ u_hat) / (n - k)

    # Finally, calculate the standard errors using the corrected covariance:
    X_pred = stats.add_constant(iv_exog)  # the second stage design matrix with a constant
    corrected_vcov = sigma_sq * np.linalg.inv(X_pred.T @ X_pred)
    corrected_se = np.sqrt(np.diag(corrected_vcov)) # the correct standard errors to pair with the beta from the IV

    # Now, just recalcuate the confidence interval of the IV beta using the fixed standard errors:
    ed_pred_index = list(iv_exog.columns).index('educ_pred') + 1 # so I can access the standard error for educ_pred
    nearc4_IV_beta = nearc4_IV_ols.fit().params['educ_pred'] # the educ_pred coefficient from the IV model
    nearc4_IV_beta_ci_low = nearc4_IV_beta - (1.96 * corrected_se[ed_pred_index])
    nearc4_IV_beta_ci_high = nearc4_IV_beta + (1.96 * corrected_se[ed_pred_index])
    nearc4_IV_beta_ci = [nearc4_IV_beta_ci_low, nearc4_IV_beta_ci_high]
    return (
        nearc4_IV_beta,
        nearc4_IV_beta_ci_high,
        nearc4_IV_beta_ci_low,
        nearc4_IV_ols,
    )


@app.cell
def _(nearc4_IV_ols):
    nearc4_IV_ols.fit(cov_type='HC0').summary()
    return


@app.cell(hide_code=True)
def _(
    lwage_ols_beta_ci,
    mo,
    nearc4_IV_beta,
    nearc4_IV_beta_ci_high,
    nearc4_IV_beta_ci_low,
):
    mo.md(f"""
    When using `nearc4` as an IV and estimating a heteroskedasticity-robust model, the coefficient of predicted education on `lwage` was {round(nearc4_IV_beta, 3)}, which had a p-value of 0.016. This would suggest that there is a positive, significant association between education and eventual wage outcomes, even when accounting for endogeneity using IV. However, the confidence interval in this model is [{round(nearc4_IV_beta_ci_low, 4)}, {round(nearc4_IV_beta_ci_high, 4)}]--compared to the interval of [{round(lwage_ols_beta_ci[0], 3)}, {round(lwage_ols_beta_ci[1], 3)}] we got in (a). This newer interval is notably less precise and also suggests that the real impact of education is smaller than the result in (a), which makes since given that we accounted for endogeneity in the model (a process which we concluded in  Question 1 will lower a model's predictive accuracy) using the proximity to college as our IV.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Part (d)
    """)
    return


@app.cell
def _(exog, raw, stats):
    # Setup another 2SLS using multiple IVs:

    # Define the relevant exogeneous data:
    both_stage_exog = stats.add_constant(exog.drop(columns=['educ'])) # exogeneous variables used in both stages

    # Since we need to evaluate whether nearc2 or nearc4 has the stronger correlation:
    educ_reduced_comparison_model = stats.OLS(
        endog=raw['educ'],
        exog=stats.add_constant(both_stage_exog.assign(nearc4=raw['nearc4'], nearc2=raw['nearc2'])), # add both IVs
    )

    # Compare model coefficients:
    educ_reduced_comparison_model.fit().summary()
    nearc4_coeff = educ_reduced_comparison_model.fit().params.loc['nearc4']
    nearc2_coeff = educ_reduced_comparison_model.fit().params.loc['nearc2']
    return both_stage_exog, nearc2_coeff, nearc4_coeff


@app.cell(hide_code=True)
def _(mo, nearc2_coeff, nearc4_coeff):
    mo.md(f"""
    When estimating the reduced form model and including `nearc2`, it's clear that being near a 2-year school has much less of relationship with years of education than proximity to a 4-year school. The coefficients are {round(nearc2_coeff, 4)} and {round(nearc4_coeff, 4)} respectively--and only the coefficient on the 4-year school proximity is statistically significant. At first glance, it seems that proximity to a 2-year school does not individually meet the relevance condition as its relationship with education--our "regressor" in the second stage regression--is not statistically significant. We will continue to estimating the 2SLS using both IVs to see if their joint presence strengthens the relationship observed in (c).
    """)
    return


@app.cell
def _(IV2SLS, both_stage_exog, raw):
    # Run the IV model with multiple IVs:
    iv_vars = raw[['nearc2', 'nearc4']] # the IVs
    multiple_ivs_model = IV2SLS(
        dependent=raw['lwage'],
        exog=both_stage_exog,
        endog=raw['educ'], # endog parameter is what we regress on the IV in the first stage
        instruments=iv_vars,
    )

    # Store the coefficient results:
    multiple_iv_beta = multiple_ivs_model.fit(cov_type='robust').params['educ']

    # Display IV results:
    multiple_ivs_model.fit(cov_type='robust')
    return (multiple_iv_beta,)


@app.cell(hide_code=True)
def _(mo, multiple_iv_beta, nearc4_IV_beta):
    mo.md(f"""
    Results with both types of college proximities from the 2SLS model are similar to what we got in (c). The coefficient increased slightly from {round(nearc4_IV_beta, 3)} to {round(multiple_iv_beta, 4)}. Statistical significance was maintained and though the confidence interval shifted outward (i.e., more positive relationship), its width changed minimally, suggesting no significant precision gains.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Part (e)
    """)
    return


@app.cell
def _(raw, stats):
    # Filter data to non-null IQ series
    subset = raw.loc[raw['IQ'] != '.'].astype({'IQ':'int64'})

    # Among the sample of non-null IQs, regress IQ on nearness to a 4-year school:
    iq_nearc4_model = stats.OLS(
        endog=subset['IQ'],
        exog=stats.add_constant(subset['nearc4']),
    )

    # Model outputs:
    iq_nearc4_beta = iq_nearc4_model.fit().params['nearc4']
    iq_nearc4_model.fit().summary()
    return iq_nearc4_beta, subset


@app.cell(hide_code=True)
def _(iq_nearc4_beta, mo):
    mo.md(f"""
    Regressing IQ on proximity to a school in an isolated OLS model results in a coefficient on `nearc4` of {round(iq_nearc4_beta, 4)}, which is statistically significant at all levels. This technically suggests that IQ is positively associated with a person's nearness to a university.
    """)
    return


@app.cell
def _(stats, subset):
    # Run a new IQ OLS model including regional covariates:
    iq_nearc4_model_w_controls = stats.OLS(
        endog=subset['IQ'],
        exog=stats.add_constant(subset[['reg669', 'reg662', 'reg661', 'smsa66', 'nearc4']]),
    )

    # Model outputs:
    iq_nearc4_control_beta = iq_nearc4_model_w_controls.fit().params['nearc4']
    iq_nearc4_model_w_controls.fit().summary()
    return (iq_nearc4_control_beta,)


@app.cell(hide_code=True)
def _(iq_nearc4_control_beta, mo):
    mo.md(f"""
    After introducing regional controls and a dummy for whether the man lived in a standard metro statistical area, the statistical significance of the coefficient on IQ was eliminated--the coefficient fell to {round(iq_nearc4_control_beta, 4)} and it had a p-value of .291. From this regression we can infer that there are region-level differences that influence IQ that were interfering in the relationship between IQ and location explored in (e). In other words, there is endogeneity in the regression of IQ on proximity to schools.  
    """)
    return


if __name__ == "__main__":
    app.run()
