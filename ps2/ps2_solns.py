import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Advanced Applied Econometrics Problem Set 2

    *Connor Martins -- 2/12/2026*
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import altair as alt
    import seaborn as so
    import statsmodels.api as stats
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats
    # from scipy import stats
    return mo, np, pd, plt, scipy_stats, so, stats


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

    I don't thinl exogeneity necessarily holds; we know that unobservable factors certainly affect wage, confounding the simple OLS model in question and introducing endogeneity. There is most likely an unobservable attribute of people in the dataset, such as an innate ability to make money, that is related to both X and Y, meaning that the error term of the OLS model will certainly be related

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
    return (ols_beta_w10,)


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
def _(mo):
    mo.md(r"""
    ### Parts (a) - (b)
    """)
    return


@app.cell
def _():
    # Latex symbols for dynamic markdown:
    x_i_la = r"$\boldsymbol{x_i}$"
    y_i_la = r"$\boldsymbol{y_i}$"
    z_i_la = r"$\boldsymbol{z_i}$"
    v_i_la = r"$\boldsymbol{v_i}$"
    eps_i_la = r"$\boldsymbol{{\epsilon}_i}$"
    exp_la = r"$\boldsymbol{E}$"
    exp_cond = r"$\boldsymbol{E[\boldsymbol{y_i}|\boldsymbol{x_i}, \boldsymbol{z_i}]}$"
    y_formula = r"$3 + \boldsymbol{x_i} + \boldsymbol{{x_i}^3} + \boldsymbol{E[\boldsymbol{{\epsilon}_i}|\boldsymbol{x_i}, \boldsymbol{z_i}]}$"
    eps_exp = r"$\boldsymbol{E[\boldsymbol{{\epsilon}_i}|\boldsymbol{x_i}, \boldsymbol{z_i}]} = 0$"
    y_formula_reduced = r"$3 + \boldsymbol{x_i} + \boldsymbol{{x_i}^3}$"
    return (
        eps_exp,
        eps_i_la,
        exp_cond,
        v_i_la,
        x_i_la,
        y_formula,
        y_formula_reduced,
        y_i_la,
        z_i_la,
    )


@app.cell
def _(
    eps_exp,
    eps_i_la,
    exp_cond,
    mo,
    v_i_la,
    x_i_la,
    y_formula,
    y_formula_reduced,
    y_i_la,
    z_i_la,
):
    mo.md(f"""
    Given that {x_i_la} is independent of {eps_i_la}, we see that, plugging in the equation for {y_i_la}:

    {exp_cond} = {y_formula}

    But we know that {eps_exp}, so we simply have:

    {exp_cond} = {y_formula_reduced}

    This expectation indicates that, since {z_i_la} is merely a function of the size of {x_i_la} added to random noise ({v_i_la}), the expectation of {y_i_la} is not dependent on {z_i_la}. Regardless of the value of {z_i_la}, the conditional expectation of {y_i_la} will be equal to zero, and therefore the causal effect is also zero.
    """)
    return


@app.cell
def _(np, stats):
    # Generate samples of estimates based on the parameters in the problem:
    vec_len_sample = 1000
    X_sample = np.random.standard_normal(vec_len_sample)
    eps_vector = np.random.standard_normal(vec_len_sample)
    X_sample_cubed = X_sample ** 3
    y_sample = np.ones(vec_len_sample)*3 + X_sample + X_sample_cubed + eps_vector

    # Construct z:
    v = np.random.normal(loc=0, scale=0.25, size=vec_len_sample)
    theta = 0.674
    z = ((X_sample + v) > theta).astype(int) # dummy for whether x+z exceeds the 75th perc. of a normal distribution
    # z

    # Regress y on z:
    y_on_z_ols = stats.OLS(
        endog=y_sample,
        exog=stats.add_constant(z)
    )

    # Check results:
    y_on_z_beta = y_on_z_ols.fit().params[1]
    return X_sample, X_sample_cubed, vec_len_sample, y_on_z_beta, y_sample, z


@app.cell(hide_code=True)
def _(mo, x_i_la, y_i_la, y_on_z_beta, y_sample, z_i_la):
    mo.md(f"""
    When I estimate the OLS of {y_i_la} on {z_i_la} alone (with an added constant), I find an OLS coefficient on {z_i_la} of {round(y_on_z_beta, 3)}, which is very large considering that {y_i_la} had a sample mean of {round(y_sample.mean(), 3)}. While this might seem like a contradiction to the preivous part, which showed a zero-causal impact between these to variables, it should be clear that this is omitted variable bias at work, given that we are clearly seeing the influence of the omitted variable, {x_i_la} in the coefficient on{z_i_la}.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Part (c)
    """)
    return


@app.cell
def _(X_sample, X_sample_cubed, np, stats, vec_len_sample, y_sample, z):
    # Regress y on x and z:
    X_sample_mat = np.column_stack([(np.ones(vec_len_sample)), X_sample, X_sample_cubed, z])
    y_on_all_ols = stats.OLS(
        endog=y_sample,
        exog=X_sample_mat
    )
    new_y_on_z_beta = y_on_all_ols.fit().params[3]
    new_y_on_z_beta
    return (new_y_on_z_beta,)


@app.cell
def _(mo, new_y_on_z_beta, y_i_la, z_i_la):
    mo.md(f"""
    After including all of the variables in the regression, the coefficient on z fell to just {round(new_y_on_z_beta,3)}, showing how including the previously omitted variable reveals the zero impact that {z_i_la} actually has on {y_i_la}. This shows a limitation of best linear predictors--without careful consideration for endogeneity issues, even satisfying the best linear predictor conditions can lead to serious issues in inference.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Question 3:
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


    def run_monte_carlo_OLS_simulation(vec_len : int, trials : int, e_dist_type : str = 'Normal'):

        ols_betas = np.zeros(trials)
        covariances = np.zeros(trials)

        # Run the simulations:
        for i in range(trials):

            # Construct the y = Xb + e elements:
            if e_dist_type == 'Normal':
                e = np.random.standard_normal(vec_len)
            elif e_dist_type == 'Logistic':
                e = np.random.logistic(loc=0, scale=1, size=vec_len)
            X_values = np.random.standard_normal(vec_len)
            ones = np.ones(vec_len) # intercept column of X
            X_mat = np.column_stack([ones, X_values])
            b = np.array([1,2])
            y_vector = (X_mat @ b) + e    

            # Run the OLS / covariance estimations:
            ols_betas[i], covariances[i] = get_ols_and_covariance_estimators(X_i=X_mat, y_i=y_vector, N=vec_len)

        return ols_betas, covariances
    return (run_monte_carlo_OLS_simulation,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Parts (a) - (b)
    """)
    return


@app.cell
def _(mo):
    # Set vector length:
    N_input = mo.ui.number(value=3, start=1, stop=100000, step=1, label='Vector Length:')
    num_trials = mo.ui.number(value=10000, start=0, stop=1000000, step=1000, label='Number of Monte-Carlo Trials:')
    error_dist_type = mo.ui.dropdown(options=['Normal', 'Logistic'], value='Normal', label='e Dist. Type:')
    mo.hstack([N_input, num_trials])
    return N_input, error_dist_type, num_trials


@app.cell
def _(N_input, num_trials, run_monte_carlo_OLS_simulation):
    # Run the simulation with the inputs:
    results = run_monte_carlo_OLS_simulation(vec_len=N_input.value, trials=num_trials.value)[0]
    return (results,)


@app.cell
def _():
    beta_one = r"$\hat{\boldsymbol{\beta_{1}}}$"
    return (beta_one,)


@app.cell
def _(beta_one, mo, num_trials):
    mo.md(f"""
    Histogram of {beta_one} over the {num_trials.value} trials:
    """)
    return


@app.cell
def _(results, so):
    # Histogram of B1 over the trials:
    so.histplot(results)
    return


@app.cell
def _(results):
    # Calculate the % of observations within beta that were >0:
    perc_over_zero = results[results > 0].shape[0] / results.shape[0]
    return (perc_over_zero,)


@app.cell(hide_code=True)
def _(mo, perc_over_zero, results):
    mo.md(f"""
    The variance of the estimates was {round(results.var(), 3)} compared to their mean of {round(results.mean(), 3)}. This high variance relative to the mean suggests a poor estimator.

    {perc_over_zero*100:.1f}% of the estimates were greater than 0.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Parts (c) - (d)
    """)
    return


@app.cell
def _(error_dist_type):
    error_dist_type
    return


@app.cell
def _(
    error_dist_type,
    np,
    num_trials,
    plt,
    run_monte_carlo_OLS_simulation,
    scipy_stats,
):
    # Plot formatting stuff:
    fig, ax = plt.subplots(figsize=(10, 6))
    vec_lengths = [5, 10, 100, 1000, 10000]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(vec_lengths)))

    # Run the simulations with increasing vector lengths:
    for length, color in zip(vec_lengths, colors):

        # Run the simulation and store the betas in an array:
        mc_betas, mc_covs = run_monte_carlo_OLS_simulation(
            vec_len=length, 
            trials=num_trials.value,
            e_dist_type=error_dist_type.value
        )

        # Calculate the t-statistics for betas associated with the current vector length N:
        t_stats = (mc_betas - 2) / np.sqrt(mc_covs) # since 2 is the true value, we use it in this formula

        # Do the t-test based on 5% alpha:
        critical_value = scipy_stats.t.ppf(0.975, df=length - 2)

        # Calculate the frequency with which we reject the null:
        rejection_rate = (np.abs(t_stats) > critical_value).mean()

        # Create one layer of the histogram:
        ax.hist(
            mc_betas,
            bins=50,
            alpha=0.5,
            color=color,
            label=f'N = {length} (Mean = {mc_betas.mean():.3f}, Variance = {mc_betas.var():.3f}, Est. Variance: {mc_covs.mean():.3f}), Rejection Rate: {rejection_rate*100:.1f}%',
            density=True
        )

    # Final plot formatting after layering is complete:
    ax.axvline(x=2, color='red', linestyle='--', linewidth=1.5, label=r'True Value of 2.0')
    ax.set_xlabel(r'$\hat{\beta}_1$')
    ax.set_ylabel('Density')
    ax.set_title(r'Distribution of OLS Coefficient by Sample Size (Monte Carlo OLS)')
    ax.legend()
    ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I can see that as I increase the size of the dataset, the distribution gets narrower (variance decreases towards zero) and the mean of the estimator gets extremely close to the true value of 2. We can see that both the mean and variance converge to their true values as we increase sample size, which is exactly what we talked about theoretically happening with the central limit theorem.

    As N increases, there is no apparent change in the rejection rate of the null hypothesis (that beta = 2), which varies between 4-6%. This means that the t-statistic is not actually related to the sample size N since the estimator itself and its standard error change in proportion with N.

    When I switch to a logistic distribution for the random error vector, the results change very little. The rejection rates are slightly higher with lower values of N, but otherwise there is little change. This shows how OLS does not require normally distributed errors, only that those errors not be endogeneous to the system (a.k.a. related to the independent variable X). Since X and e are still generated randomly and independently even after the distribution change, the OLS estimator still converges in probability to the true value.
    """)
    return


if __name__ == "__main__":
    app.run()
