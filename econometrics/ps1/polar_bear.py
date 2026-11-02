import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from scipy import stats
    return np, pd, stats


@app.cell
def _(pd):
    # Read polar bear data:
    df = pd.read_csv('boston_lowest_temp.csv')
    df
    return (df,)


@app.cell
def _(df, np, stats):
    ### Part 1:

    # Calculate sample mean / variance:
    X_bar = df["lowest_temp_F"].mean()
    S_squared = df["lowest_temp_F"].var()
    print(f'Sample Mean: {X_bar}')
    print(f'Sample Variance: {S_squared}')

    # Calculate prediction interval
    n=len(df)
    t_crit = stats.t.ppf(0.975, df=n-1)  # 95% -> 0.975 upper tail
    margin = t_crit * np.sqrt(S_squared) * np.sqrt(1 + 1/n)
    lower = X_bar - margin
    upper = X_bar + margin
    print(f'Prediction Interval: [{lower}, {upper}]')
    return S_squared, X_bar, n, t_crit


@app.cell
def _(S_squared, X_bar, n, np, t_crit):
    ### Part 2:

    # Calculate the CI of the expectation of X:
    ci_margin = t_crit * (np.sqrt(S_squared) / np.sqrt(n))
    ci_lower = X_bar - ci_margin
    ci_upper = X_bar + ci_margin
    print(f'Confidence Interval of E[X]: [{ci_lower}, {ci_upper}]')
    return


@app.cell
def _(df):
    ### Part 3:

    # Calculate sample means and SDs of two halves of the data:
    df1 = df.loc[df['year']<=1951]
    df2 = df.loc[df['year']>1951]
    first_mean=df1['lowest_temp_F'].mean()
    second_mean=df2['lowest_temp_F'].mean()
    first_sd=df1['lowest_temp_F'].std()
    second_sd=df2['lowest_temp_F'].std()
    first_S=df1['lowest_temp_F'].var()
    second_S=df2['lowest_temp_F'].var()

    print(f'Sample 1: Mean {first_mean}; SD {first_sd}')
    print(f'Sample 2: Mean {second_mean}; SD {second_sd}')
    return df1, df2, first_S, first_mean, second_S, second_mean


@app.cell
def _(df1, df2, first_S, first_mean, np, second_S, second_mean, stats):
    ### T test of sample means:
    n1 = len(df1)
    n2 = len(df2)

    # Pooled variance formula
    S_pooled_sq = ((n1 - 1) * first_S + (n2 - 1) * second_S) / (n1 + n2 - 2)
    S_pooled = np.sqrt(S_pooled_sq)
    # S_pooled

    # Test statistic
    t_stat = (first_mean - second_mean) / (S_pooled * np.sqrt(1/n1 + 1/n2))
    # t_stat

    # Degrees of freedom
    deg_free = n1 + n2 - 2

    # Two-sided p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), deg_free))
    p_value
    return


if __name__ == "__main__":
    app.run()
