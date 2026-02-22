import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Advanced Applied Econometrics Problem Set 3

    *Connor Martins -- 2/19/2026*
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Question 3:
    """)
    return


@app.cell
def _():
    # Latex symbols:
    xi_la = r"$\boldsymbol{x_i}$"
    yi_la = r"$\boldsymbol{y_i}$"
    zi_la = r"$\boldsymbol{z_i}$"
    vi_la = r"$\boldsymbol{v_i}$"
    exp_uv = r"$\boldsymbol{E[\boldsymbol{u_i}|\boldsymbol{v_i}]}$"
    beta_la = r"$\boldsymbol{\beta}$"
    return beta_la, exp_uv, xi_la


@app.cell
def _(beta_la, exp_uv, mo):
    mo.md(f"""
    Part (a)

    Note that {exp_uv} = 0.9 and {beta_la} = 1.
    """)
    return


@app.cell
def _(np):
    # Prepare a custom function which returns an OLS estimator:
    def get_ols_and_covariance_estimators(X_i, y_i, N):

        # Create the OLS formula terms using numpy:
        XtX_inv = np.linalg.inv(X_i.T @ X_i)
        b_hat = XtX_inv @ X_i.T @ y_i
        resid = y_i - X_i @ b_hat
        s2 = (resid @ resid) / (N - 2)
        cov_est = s2 * XtX_inv[1, 1]
        return b_hat[1], cov_est


    def run_monte_carlo_OLS_IV_simulation(vec_len : int, trials : int, pi_val = int):

        # Initialize list of model coefficients:
        first_stage_t_stats = np.zeros(trials)
        ols_betas = np.zeros(trials)
        covariances = np.zeros(trials)
        sec_stage_t_stats = np.zeros(trials)

        # Create covariance matrix of  ui and vi according to the problem assumption:
        u_v_cov_mat = np.array([
            [1, 0.9],
            [0.9, 1]
        ])

        # Run the simulations:
        for i in range(trials):

            # Create random ui and vi vectors from their multivariate distribution:
            ui_vi_matrix = np.random.multivariate_normal(mean=[0, 0], cov=u_v_cov_mat, size=vec_len)
            ui, vi = ui_vi_matrix[:, 0], ui_vi_matrix[:, 1]

            # First, use zi and pi to generate xi:
            zi = np.random.standard_normal(vec_len)
            pi_vec = np.ones(vec_len) * pi_val
            xi = (zi * pi_vec) + vi

            # Now, run the first stage regression: regress x on z to recover pi:
            ones = np.ones(vec_len) # intercept column of Z
            Z_mat = np.column_stack([ones, zi])
            ZtZ_inv = np.linalg.inv(Z_mat.T @ Z_mat)
            pi_hat = ZtZ_inv @ Z_mat.T @ xi # here's where we estimate pi hat

            # Estimate covariance from the first stage regression to calculate t-stat:
            pi_var = get_ols_and_covariance_estimators(
                X_i=Z_mat, 
                y_i=xi, 
                N=vec_len
            )[1]
            first_stage_t_stats[i] = pi_hat[1] / np.sqrt(pi_var)

            # Calculate x hat using pi hat:
            xi_hat = Z_mat @ pi_hat # predict xi from pi hat

            # Now, run the second stage: regress y on x hat:
            yi = xi + ui # this assumes Beta = 1
            X_hat_mat = np.column_stack([ones, xi_hat])
            sec_stage_beta, sec_stage_var = get_ols_and_covariance_estimators(X_i=X_hat_mat, y_i=yi, N=vec_len)
            sec_stage_t_stat = sec_stage_beta / np.sqrt(sec_stage_var) # t-stat for the 2nd stage coefficient
            ols_betas[i], covariances[i], sec_stage_t_stats[i] = sec_stage_beta, sec_stage_var, sec_stage_t_stat

        return ols_betas, covariances, first_stage_t_stats, sec_stage_t_stats


    #### SIMULATION EXERCISE:

    # Establish testing levels of pi:
    pis = [0, .25, 1, 2]

    # Set numer of observations:
    n_obs = 100

    # Simulate 1,000 times (vectorized) over each value of pi:
    betas = []
    covs = []
    first_stage_t_stats = []
    rejection_rates = []
    for pi in pis:

        # Conduct 1,000 trials with the in-loop level of pi:
        pi_betas, pi_covs, pi_t_stats, sec_t_stats = run_monte_carlo_OLS_IV_simulation(trials=1000, vec_len=n_obs, pi_val=pi)
        betas.append(pi_betas)
        first_stage_t_stats.append(pi_t_stats)

        # Calc the null hypothesis rejection rate based on the t-statistic from the first stage:
        pi_rejection_rate = (np.abs(pi_t_stats) > 1.96).mean()
        rejection_rates.append(pi_rejection_rate)
    return (
        betas,
        first_stage_t_stats,
        n_obs,
        pis,
        rejection_rates,
        run_monte_carlo_OLS_IV_simulation,
    )


@app.cell
def _(first_stage_t_stats, pis):
    # Compare distributions of first stage t-stats visually:
    import matplotlib.pyplot as plt

    # Loop over the different values of pi and plot the distributions of the first stage regression t-stat:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for idx, (ax, _pi, t_stats) in enumerate(zip(axes, pis, first_stage_t_stats)):
        ax.hist(t_stats, bins=40, edgecolor='black', alpha=0.7)
        ax.set_title(f'π = {_pi}')
        ax.set_xlabel('t-statistic')
        ax.axvline(x=1.96, color='red', linestyle='--', label='±1.96')
        ax.axvline(x=-1.96, color='red', linestyle='--')
        if idx == 0:
            ax.set_ylabel('Frequency')

    fig.suptitle('First Stage t-statistic Distributions by π', fontsize=14)
    plt.tight_layout()
    ax
    return (plt,)


@app.cell(hide_code=True)
def _(mo, rejection_rates):
    mo.md(f"""
    From the above plot, we can see that as we increase pi, the distribution of t-statistics of the first-stage regression shifts away from zero, which is what we want because this indicates a strengthening of the relevance condition. The rejection rate goes from {rejection_rates[0]*100:.2f}% at pi = 0 to {rejection_rates[3]*100:.2f}% with pi = 2, indicating that the higher pi is, the better instrument it is. The more different that the first stage estimator is from zero, the more related our regressor is to the instrument, and the better our second stage regressionn will perform.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Part (b)
    """)
    return


@app.cell
def _(betas, np, pis, plt):
    # Loop over the different values of pi and plot the distributions of the IV estimator:
    _fig, _axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for _idx, (_ax, _pi, iv_betas) in enumerate(zip(_axes, pis, betas)):
        _sim_beta_mean = round(iv_betas.mean(), 2)
        _sim_beta_median = round(np.median(iv_betas), 2)
        _ax.hist(iv_betas, bins=40, edgecolor='black', alpha=0.7)
        _ax.set_title(f'π = {_pi}; (Mean = {_sim_beta_mean}, Median = {_sim_beta_median})')
        _ax.set_xlabel('t-statistic')
        _ax.axvline(x=1, color='red', linestyle='--', label='True Value: 1')
        if _idx == 0:
            _ax.set_ylabel('Frequency')

    _fig.suptitle(f'IV Estimators Distributions by π', fontsize=14)
    _ax
    return


@app.cell(hide_code=True)
def _(mo, xi_la):
    mo.md(f"""
    As pi increases, the distribution of the IV estimator shifts closer to the true value of 1 and gets narrower, indicating improved accuracy of the IV setup as the IV gets stronger (i.e., more related to {xi_la}). The mean / median value of the estimator gets closer to one as pi increases, while the rejection rate goes to 100%.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Part (c)
    """)
    return


@app.cell
def _(n_obs, np, pis, plt, run_monte_carlo_OLS_IV_simulation):
    # Same monte carlo trial setup, but only include if the t-statistic of the first stage is sufficiently large:
    _betas = []
    _covs = []
    _first_stage_t_stats = []
    _rejection_rates = []
    for _pi in pis:

        # Conduct 1,000 trials with the in-loop level of pi:
        _pi_betas, _pi_covs, _pi_t_stats, _pi_sec_t_stats = run_monte_carlo_OLS_IV_simulation(
            trials=1000, 
            vec_len=n_obs, 
            pi_val=_pi
        )

        # Create a filter condition based on first stage t-stat value:
        t_stat_mask = np.abs(_pi_t_stats) > 1.96

        # Append filtered versions of the betas / t-stats: 
    
        _betas.append(_pi_betas[t_stat_mask])
        _first_stage_t_stats.append(_pi_t_stats[t_stat_mask])

        # Calc the null hypothesis rejection rate based on the t-statistic from the SECOND stage:
        _pi_sec_rejection_rate = (np.abs(_pi_sec_t_stats[t_stat_mask]) > 1.96).mean() # only where first stage t-stat > 1.96
        _rejection_rates.append(_pi_sec_rejection_rate)


    # Loop over the different values of pi and plot the distributions of the IV estimator:
    _fig, _axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for _idx, (_ax, _pi, _iv_betas, _sec_rej_rate) in enumerate(zip(_axes, pis, _betas, _rejection_rates)):
        _sim_beta_mean = round(_iv_betas.mean(), 2) # mean value of the second stage regressor
        _sim_beta_median = round(np.median(_iv_betas), 2) # median value of the second stage regressor
        _ax.hist(_iv_betas, bins=40, edgecolor='black', alpha=0.7)
        _ax.set_title(f'π = {_pi}; (IV Rej. Rate: {_sec_rej_rate*100:.1f}%)')
        _ax.set_xlabel('t-statistic')
        _ax.axvline(x=1, color='red', linestyle='--', label='True Value: 1')
        if _idx == 0:
            _ax.set_ylabel('Frequency')

    _fig.suptitle(f'IV Estimators Distributions by π with First Stage T-Stat > 1,96 Condition', fontsize=14)
    _ax
    return


@app.cell(hide_code=True)
def _(mo, xi_la):
    mo.md(f"""
    As shown by the chart above, when only including cases which had a first stage regression meet the relevance IV condition (so the first stage t-stat > 1.96), the distributions of the IV estimators tend to shift away from zero even at smaller levels of pi. This is intuitive as we eliminated trials which produced results that failed to satisfy the relevance condition. We basically artificially strengthened the IV by only selecting instances in which it was most correlated with our {xi_la} term, thus driving improvements in the size of the 5% tests on the IV estimators themselves. This is shown most clearly in the pi = 0 case, when the size of the test on the IV estimator was over 80% despite the poor results we noted at this level in (b). This shows that the relevance condition can actualy bias results because if we run 2SLS purely based on this condition. It could lead us to use a weak IV because the selection bias at play (i.e., only including the instances of strong correlation between the IV and the endogeneous regressor) makes the IV appear stronger than it actually is.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Part (d)
    """)
    return


@app.cell
def _(n_obs, np, pis, plt, run_monte_carlo_OLS_IV_simulation):
    # Same monte carlo trial setup, but only include if the first stage t-stat squared > 10::
    _betas = []
    _covs = []
    _first_stage_t_stats = []
    _rejection_rates = []
    for _pi in pis:

        # Conduct 1,000 trials with the in-loop level of pi:
        _pi_betas, _pi_covs, _pi_t_stats, _pi_sec_t_stats = run_monte_carlo_OLS_IV_simulation(
            trials=1000, 
            vec_len=n_obs, 
            pi_val=_pi
        )

        # Create a filter condition based on first stage t-stat value:
        f_stat_mask = np.abs(_pi_t_stats ** 2) > 10

        # Append filtered versions of the betas / t-stats: 
    
        _betas.append(_pi_betas[f_stat_mask])
        _first_stage_t_stats.append(_pi_t_stats[f_stat_mask])

        # Calc the null hypothesis rejection rate based on the t-statistic from the SECOND stage:
        _pi_sec_rejection_rate = (np.abs(_pi_sec_t_stats[f_stat_mask]) > 1.96).mean() # only where first stage t-stat > 1.96
        _rejection_rates.append(_pi_sec_rejection_rate)


    # Loop over the different values of pi and plot the distributions of the IV estimator:
    _fig, _axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for _idx, (_ax, _pi, _iv_betas, _sec_rej_rate) in enumerate(zip(_axes, pis, _betas, _rejection_rates)):
        _sim_beta_mean = round(_iv_betas.mean(), 2) # mean value of the second stage regressor
        _sim_beta_median = round(np.median(_iv_betas), 2) # median value of the second stage regressor
        _ax.hist(_iv_betas, bins=40, edgecolor='black', alpha=0.7)
        _ax.set_title(f'π = {_pi}; (IV Rej. Rate: {_sec_rej_rate*100:.1f}%)')
        _ax.set_xlabel('t-statistic')
        _ax.axvline(x=1, color='red', linestyle='--', label='True Value: 1')
        if _idx == 0:
            _ax.set_ylabel('Frequency')

    _fig.suptitle(f'IV Estimators Distributions by π with First Stage T-Stat > 1,96 Condition', fontsize=14)
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As shown in the chart above, the F-stat condition improved the power of the tests on the IV estimator, moving the rejection rate to 100% in the pi = 0 group. This exacerbated the selection bias we saw in (c) because only a very small sample of the pi = 0 simulations made it through to the second stage, most of which ended up producing significant IV estimators.
    """)
    return


if __name__ == "__main__":
    app.run()
