import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Advanced Applied Econometrics Problem Set 4

    *Connor Martins -- 2/12/2026*
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
    # Question 2
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Load Data:
    """)
    return


@app.cell
def _(pd):
    data = pd.read_excel('jtrain3.xlsx')
    data
    return (data,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Part (a)
    """)
    return


@app.cell
def _(data, stats):
    # Estimate an OLS model to generate ATE:
    simple_ate_model = stats.OLS(
        endog=data['unem78'],
        exog=stats.add_constant(data['train']),
    )

    # Extract the coefficient:
    train_ate = simple_ate_model.fit(cov_type='HC0').params['train']
    return simple_ate_model, train_ate


@app.cell
def _(simple_ate_model):
    simple_ate_model.fit().summary()
    return


@app.cell(hide_code=True)
def _(mo, train_ate):
    mo.md(f"""
    The OLS model on `train` shows an average treatment effect of {round(train_ate, 3)}, suggesting that having received the training program is associated with a {round(train_ate, 3)} percentage point increase in likelihood of being unemployed in 1978. This would suggest the opposite of the desired program effect--it increases unemployment likelihood by a significant margin.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Part (b):
    """)
    return


@app.cell
def _(data, stats):
    # Get controls data:
    control_cols = ['age', 'educ', 'black', 'hisp', 'married', 're74', 're75', 'unem75', 'unem74']
    control_data = data[control_cols + ['train']]

    # Re-run the OLS 
    simple_ate_model_w_controls = stats.OLS(
        endog=data['unem78'],
        exog=stats.add_constant(control_data),
    )

    # Extract the coefficient:
    train_ate_w_controls = simple_ate_model_w_controls.fit().params['train']
    return control_data, simple_ate_model_w_controls, train_ate_w_controls


@app.cell
def _(simple_ate_model_w_controls):
    simple_ate_model_w_controls.fit(cov_type='HC0').summary()
    return


@app.cell
def _(mo, simple_ate_model_w_controls, train_ate_w_controls):
    mo.md(rf"""
    The ATE from the model with controls is {round(train_ate_w_controls, 3)}, suggesting that the program had the desired effect of reducing the probability that a person is unemployed in 1978 if they received the training when including control variables like race, education, and racial identity. The robust 95% confidence interval for this coefficient is [{round(simple_ate_model_w_controls.fit().conf_int().loc['train'][0], 3)}, {round(simple_ate_model_w_controls.fit().conf_int().loc['train'][1], 3)}].
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Part (c):
    """)
    return


@app.cell
def _(control_data, data, stats):
    # Estimate an OLS model separately for the treatment and control groups:
    treatment_ate_model = stats.OLS(
        endog=data.query("train==1")['unem78'],
        exog=stats.add_constant(control_data.query("train==1")),
    ).fit(cov_type='HC0')

    control_ate_model = stats.OLS(
        endog=data.query("train==0")['unem78'],
        exog=stats.add_constant(control_data.query("train==0")),
    ).fit(cov_type='HC0')

    # Generate predictions of the outcome variable using each model:
    treatment_pred = treatment_ate_model.predict(stats.add_constant(control_data.query("train==1")))
    control_pred = control_ate_model.predict(stats.add_constant(control_data.query("train==0")))
    average_diff = treatment_pred.mean() - control_pred.mean()
    average_diff
    return (average_diff,)


@app.cell
def _(average_diff, mo):
    mo.md(f"""
    The estimate using the average of the predictions of the separate models is {round(average_diff, 3)}--identical to the result from part (a).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Part (d):
    """)
    return


@app.cell
def _(control_data, data, stats):
    # Estimate an OLS model separately for the treatment and control groups but only use
    # people unemployed in 1974 or 1975:

    _control_data = control_data.query("(unem74==1) | (unem75==1)")

    _treatment_ate_model = stats.OLS(
        endog=(
            data
            .query("(unem74==1) | (unem75==1)")
            .query("train==1")
            ['unem78']
        ),
        exog=stats.add_constant(_control_data.query("train==1")),
    )

    _control_ate_model = stats.OLS(
        endog=(
            data
            .query("(unem74==1) | (unem75==1)")
            .query("train==0")
            ['unem78']
        ),
        exog=stats.add_constant(_control_data.query("train==0")),
    )

    # Generate predictions of the outcome variable using each model:
    _treatment_pred = _treatment_ate_model.fit(cov_type='HC0').predict(stats.add_constant(_control_data.query("train==1")))
    _control_pred = _control_ate_model.fit(cov_type='HC0').predict(stats.add_constant(_control_data.query("train==0")))

    # Take the difference of the prediction averages to derive the ATE:
    filtered_average_diff = _treatment_pred.mean() - _control_pred.mean()
    filtered_average_diff
    return (filtered_average_diff,)


@app.cell(hide_code=True)
def _(filtered_average_diff, mo):
    mo.md(f"""
    Repeating the procedure from (c) with only those people unemployed in either 1974 or 1975, we get an ATE of {round(filtered_average_diff, 3)}--suggesting that the program has the intended effect for the unemployed since it significantly reduces the probability of being unemployed in 1978 for those who went through the job training program. Compared to our model in (c), which generated the same positive coefficient as in (a), this model suggests that program might benefit the previously unemployed more than the previously employed.
    """)
    return


if __name__ == "__main__":
    app.run()
