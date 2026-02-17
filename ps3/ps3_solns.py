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
    return (mo,)


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


if __name__ == "__main__":
    app.run()
