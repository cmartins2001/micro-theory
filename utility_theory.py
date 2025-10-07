"""
Marimo Notebook: Utility Theory and Consumer Choice
Economics Learning Environment
Author: [Your Name]
Course: Micro Theory - Fall 2024

This notebook explores utility maximization, budget constraints, 
and comparative statics using reactive programming.

To run: marimo edit 01_utility_theory.py
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="full")


@app.cell
def __():
    """Cell 1: Imports and Setup"""
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from scipy.optimize import minimize, NonlinearConstraint
    import plotly.graph_objects as go
    from dataclasses import dataclass
    from typing import Callable, Dict, Tuple, Optional
    
    # Set plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    mo.md("""
    # Utility Theory and Consumer Choice
    
    ## Learning Objectives
    1. Understand utility functions and their properties
    2. Solve constrained optimization problems analytically and numerically
    3. Visualize indifference curves and budget constraints
    4. Perform comparative statics analysis
    5. Practice LaTeX typesetting of economic problems
    """)
    return (
        Callable,
        Dict,
        NonlinearConstraint,
        Optional,
        Tuple,
        cm,
        dataclass,
        go,
        minimize,
        mo,
        np,
        pd,
        plt,
    )


@app.cell
def __(Callable, Dict, Optional, Tuple, dataclass):
    """Cell 2: Data Structures for Economic Problems"""
    
    @dataclass
    class UtilityMaxProblem:
        """Represents a utility maximization problem"""
        utility_func: Callable
        px: float
        py: float
        income: float
        alpha: Optional[float] = None  # For Cobb-Douglas
        rho: Optional[float] = None    # For CES
        name: str = "Utility Maximization"
        
        def budget_constraint(self, x: float, y: float) -> float:
            """Returns budget constraint: px*x + py*y"""
            return self.px * x + self.py * y
        
        def is_feasible(self, x: float, y: float) -> bool:
            """Check if bundle (x,y) satisfies budget constraint"""
            return self.budget_constraint(x, y) <= self.income + 1e-6
        
        def __repr__(self):
            return f"{self.name}(px={self.px}, py={self.py}, I={self.income})"
    
    @dataclass
    class Solution:
        """Stores solution to optimization problem"""
        x_star: float
        y_star: float
        utility_star: float
        lagrange_multiplier: Optional[float] = None
        success: bool = True
        message: str = "Optimization successful"
        
        def to_dict(self) -> Dict:
            return {
                'x*': self.x_star,
                'y*': self.y_star,
                'U*': self.utility_star,
                'λ': self.lagrange_multiplier if self.lagrange_multiplier else 'N/A'
            }
    
    return Solution, UtilityMaxProblem


@app.cell
def __(mo):
    """Cell 3: Interactive Controls - Function Type Selection"""
    
    function_selector = mo.ui.dropdown(
        options={
            'Cobb-Douglas: U(x,y) = x^α y^(1-α)': 'cobb_douglas',
            'CES: U(x,y) = [αx^ρ + (1-α)y^ρ]^(1/ρ)': 'ces',
            'Perfect Substitutes: U(x,y) = ax + by': 'perfect_sub',
            'Perfect Complements: U(x,y) = min{ax, by}': 'leontief',
            'Quasilinear: U(x,y) = ln(x) + y': 'quasilinear'
        },
        value='cobb_douglas',
        label="Select Utility Function"
    )
    
    mo.md(f"""
    ## Step 1: Choose Utility Function
    
    Select the type of utility function to analyze:
    """).center()
    return (function_selector,)


@app.cell
def __(function_selector, mo):
    """Cell 4: Display Function Selector"""
    function_selector
    return


@app.cell
def __(function_selector, mo):
    """Cell 5: Parameter Controls (Reactive to Function Type)"""
    
    # Base parameters that all functions need
    px_slider = mo.ui.slider(
        start=0.5, stop=10, step=0.5, value=2, 
        label="Price of Good X (px)", show_value=True
    )
    
    py_slider = mo.ui.slider(
        start=0.5, stop=10, step=0.5, value=3,
        label="Price of Good Y (py)", show_value=True
    )
    
    income_slider = mo.ui.slider(
        start=20, stop=300, step=10, value=120,
        label="Income (I)", show_value=True
    )
    
    # Function-specific parameters
    alpha_slider = mo.ui.slider(
        start=0.1, stop=0.9, step=0.05, value=0.6,
        label="α (preference weight for X)", show_value=True
    )
    
    rho_slider = mo.ui.slider(
        start=-2, stop=0.9, step=0.1, value=0.5,
        label="ρ (substitution parameter)", show_value=True
    )
    
    a_slider = mo.ui.slider(
        start=0.5, stop=5, step=0.5, value=2,
        label="a (coefficient on X)", show_value=True
    )
    
    b_slider = mo.ui.slider(
        start=0.5, stop=5, step=0.5, value=1,
        label="b (coefficient on Y)", show_value=True
    )
    
    # Show different parameters based on function type
    if function_selector.value == 'cobb_douglas':
        param_controls = mo.vstack([
            mo.md("### Price and Income Parameters"),
            px_slider, py_slider, income_slider,
            mo.md("### Utility Function Parameters"),
            alpha_slider
        ])
    elif function_selector.value == 'ces':
        param_controls = mo.vstack([
            mo.md("### Price and Income Parameters"),
            px_slider, py_slider, income_slider,
            mo.md("### CES Parameters"),
            alpha_slider, rho_slider,
            mo.md("*Note: ρ → 0 gives Cobb-Douglas, ρ → -∞ gives Leontief*")
        ])
    elif function_selector.value in ['perfect_sub', 'leontief']:
        param_controls = mo.vstack([
            mo.md("### Price and Income Parameters"),
            px_slider, py_slider, income_slider,
            mo.md("### Utility Function Coefficients"),
            a_slider, b_slider
        ])
    else:  # quasilinear
        param_controls = mo.vstack([
            mo.md("### Price and Income Parameters"),
            px_slider, py_slider, income_slider
        ])
    
    mo.md("## Step 2: Set Parameters").center()
    return (
        a_slider,
        alpha_slider,
        b_slider,
        income_slider,
        param_controls,
        px_slider,
        py_slider,
        rho_slider,
    )


@app.cell
def __(param_controls):
    """Cell 6: Display Parameter Controls"""
    param_controls
    return


@app.cell
def __(
    UtilityMaxProblem,
    a_slider,
    alpha_slider,
    b_slider,
    function_selector,
    income_slider,
    mo,
    np,
    px_slider,
    py_slider,
    rho_slider,
):
    """Cell 7: Define Utility Function and Create Problem (Reactive)"""
    
    # Get current parameter values
    _px = px_slider.value
    _py = py_slider.value
    _income = income_slider.value
    _alpha = alpha_slider.value
    _rho = rho_slider.value
    _a = a_slider.value
    _b = b_slider.value
    
    # Define utility function based on selection
    if function_selector.value == 'cobb_douglas':
        utility_func = lambda x, y: x**_alpha * y**(1-_alpha)
        problem_latex = (
            r"U(x,y) = x^{" + f"{_alpha:.2f}" + r"} y^{" + f"{1-_alpha:.2f}" + r"}"
        )
        analytical_solution = {
            'x*': (_alpha * _income) / _px,
            'y*': ((1-_alpha) * _income) / _py
        }
        problem_name = "Cobb-Douglas Utility"
        
    elif function_selector.value == 'ces':
        utility_func = lambda x, y: (_alpha * x**_rho + (1-_alpha) * y**_rho)**(1/_rho) if _rho != 0 else x**_alpha * y**(1-_alpha)
        problem_latex = (
            r"U(x,y) = \left[" + f"{_alpha:.2f}" + r" x^{" + f"{_rho:.2f}" + 
            r"} + " + f"{1-_alpha:.2f}" + r" y^{" + f"{_rho:.2f}" + r"}\right]^{" +
            f"{1/_rho:.2f}" + r"}"
        )
        analytical_solution = None  # CES requires numerical solution
        problem_name = "CES Utility"
        
    elif function_selector.value == 'perfect_sub':
        utility_func = lambda x, y: _a * x + _b * y
        problem_latex = f"U(x,y) = {_a:.1f}x + {_b:.1f}y"
        # Corner solution
        if _a/_px > _b/_py:
            analytical_solution = {'x*': _income/_px, 'y*': 0}
        else:
            analytical_solution = {'x*': 0, 'y*': _income/_py}
        problem_name = "Perfect Substitutes"
        
    elif function_selector.value == 'leontief':
        utility_func = lambda x, y: np.minimum(_a * x, _b * y)
        problem_latex = f"U(x,y) = \\min\\{{{_a:.1f}x, {_b:.1f}y\\}}"
        analytical_solution = {
            'x*': _income / (_px + _py * _a/_b),
            'y*': _income / (_py + _px * _b/_a)
        }
        problem_name = "Perfect Complements"
        
    else:  # quasilinear
        utility_func = lambda x, y: np.log(x + 1e-10) + y
        problem_latex = r"U(x,y) = \ln(x) + y"
        analytical_solution = {
            'x*': _py / _px,
            'y*': (_income - _px * (_py/_px)) / _py
        }
        problem_name = "Quasilinear Utility"
    
    # Create problem instance
    problem = UtilityMaxProblem(
        utility_func=utility_func,
        px=_px,
        py=_py,
        income=_income,
        alpha=_alpha if function_selector.value in ['cobb_douglas', 'ces'] else None,
        rho=_rho if function_selector.value == 'ces' else None,
        name=problem_name
    )
    
    mo.md("Problem defined successfully ✓").callout(kind="success")
    return (
        analytical_solution,
        problem,
        problem_latex,
        problem_name,
        utility_func,
    )


@app.cell
def __(mo, problem, problem_latex):
    """Cell 8: Display Problem Statement in LaTeX"""
    
    mo.md(f"""
    ## Step 3: Problem Formulation
    
    ### Utility Maximization Problem
    
    $$
    \\begin{{aligned}}
    \\max_{{x,y}} \\quad & {problem_latex} \\\\
    \\text{{subject to}} \\quad & {problem.px}x + {problem.py}y = {problem.income} \\\\
    & x, y \\geq 0
    \\end{{aligned}}
    $$
    
    ### Economic Interpretation
    
    The consumer seeks to maximize utility subject to their budget constraint.
    The Lagrangian for this problem is:
    
    $$
    \\mathcal{{L}}(x, y, \\lambda) = {problem_latex} + \\lambda({problem.income} - {problem.px}x - {problem.py}y)
    $$
    
    ### First-Order Conditions (FOCs)
    
    $$
    \\begin{{aligned}}
    \\frac{{\\partial \\mathcal{{L}}}}{{\\partial x}} &= 0 \\\\
    \\frac{{\\partial \\mathcal{{L}}}}{{\\partial y}} &= 0 \\\\
    \\frac{{\\partial \\mathcal{{L}}}}{{\\partial \\lambda}} &= 0
    \\end{{aligned}}
    $$
    """)
    return


@app.cell
def __(
    NonlinearConstraint,
    Solution,
    minimize,
    problem,
    utility_func,
):
    """Cell 9: Numerical Solution Using SciPy"""
    
    # Objective function (negative for minimization)
    def _objective(vars):
        x, y = vars
        if x <= 0 or y <= 0:
            return 1e10  # Penalty for infeasible points
        return -utility_func(x, y)
    
    # Budget constraint
    _constraint = NonlinearConstraint(
        lambda vars: problem.px * vars[0] + problem.py * vars[1],
        problem.income,
        problem.income
    )
    
    # Bounds
    _bounds = [(1e-6, problem.income/problem.px), (1e-6, problem.income/problem.py)]
    
    # Initial guess
    _x0 = [problem.income/(2*problem.px), problem.income/(2*problem.py)]
    
    # Solve
    _result = minimize(
        _objective,
        _x0,
        method='SLSQP',
        constraints=_constraint,
        bounds=_bounds,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    
    # Extract solution
    if _result.success:
        numerical_solution = Solution(
            x_star=_result.x[0],
            y_star=_result.x[1],
            utility_star=-_result.fun,
            success=True,
            message=_result.message
        )
    else:
        numerical_solution = Solution(
            x_star=0, y_star=0, utility_star=0,
            success=False,
            message=_result.message
        )
    
    # Verify budget constraint
    _budget_check = problem.px * numerical_solution.x_star + problem.py * numerical_solution.y_star
    _budget_satisfied = abs(_budget_check - problem.income) < 0.01
    
    del _objective, _constraint, _bounds, _x0, _result, _budget_check
    return _budget_satisfied, numerical_solution


@app.cell
def __(_budget_satisfied, analytical_solution, mo, numerical_solution):
    """Cell 10: Display Results"""
    
    results_display = mo.vstack([
        mo.md("## Step 4: Solution Results"),
        
        mo.hstack([
            mo.stat(
                value=f"{numerical_solution.x_star:.4f}",
                label="Optimal x*",
                caption="Units of good X",
                bordered=True
            ),
            mo.stat(
                value=f"{numerical_solution.y_star:.4f}",
                label="Optimal y*",
                caption="Units of good Y",
                bordered=True
            ),
            mo.stat(
                value=f"{numerical_solution.utility_star:.4f}",
                label="Maximum Utility U*",
                caption="Utils",
                bordered=True
            ),
        ]),
        
        mo.callout(
            mo.md(f"✓ Budget constraint satisfied: {_budget_satisfied}"),
            kind="success" if _budget_satisfied else "warn"
        ),
        
        # Analytical vs Numerical comparison if available
        mo.md("### Analytical vs Numerical Solution") if analytical_solution else mo.md(""),
        
        mo.ui.table(
            pd.DataFrame({
                'Variable': ['x*', 'y*'],
                'Analytical': [f"{analytical_solution.get('x*', 'N/A'):.4f}" if analytical_solution and analytical_solution.get('x*') else 'N/A',
                              f"{analytical_solution.get('y*', 'N/A'):.4f}" if analytical_solution and analytical_solution.get('y*') else 'N/A'],
                'Numerical': [f"{numerical_solution.x_star:.4f}", f"{numerical_solution.y_star:.4f}"],
                'Difference': [f"{abs(analytical_solution.get('x*', 0) - numerical_solution.x_star):.6f}" if analytical_solution and analytical_solution.get('x*') else 'N/A',
                              f"{abs(analytical_solution.get('y*', 0) - numerical_solution.y_star):.6f}" if analytical_solution and analytical_solution.get('y*') else 'N/A']
            })
        ) if analytical_solution else mo.md("*Analytical solution not available for this utility function*")
    ])
    
    results_display
    return (results_display,)


@app.cell
def __(mo, np, numerical_solution, plt, problem, utility_func):
    """Cell 11: 2D Visualization - Indifference Curves"""
    
    # Create meshgrid
    _x_range = (0, min(problem.income/problem.px * 1.2, 100))
    _y_range = (0, min(problem.income/problem.py * 1.2, 100))
    
    _x = np.linspace(_x_range[0] + 0.1, _x_range[1], 300)
    _y = np.linspace(_y_range[0] + 0.1, _y_range[1], 300)
    _X, _Y = np.meshgrid(_x, _y)
    
    # Compute utility
    try:
        _Z = utility_func(_X, _Y)
    except:
        _Z = np.zeros_like(_X)
        for i in range(_X.shape[0]):
            for j in range(_X.shape[1]):
                try:
                    _Z[i,j] = utility_func(_X[i,j], _Y[i,j])
                except:
                    _Z[i,j] = np.nan
    
    # Create figure
    fig2d, ax2d = plt.subplots(figsize=(12, 9))
    
    # Plot indifference curves
    _contour = ax2d.contour(_X, _Y, _Z, levels=20, cmap='viridis', alpha=0.6)
    ax2d.clabel(_contour, inline=True, fontsize=8)
    
    # Plot budget line
    _x_budget = np.linspace(0, problem.income/problem.px, 100)
    _y_budget = (problem.income - problem.px * _x_budget) / problem.py
    ax2d.plot(_x_budget, _y_budget, 'r-', linewidth=3, 
             label=f'Budget: ${problem.px:.1f}x + ${problem.py:.1f}y = ${problem.income:.0f}$')
    
    # Plot optimal point
    ax2d.plot(numerical_solution.x_star, numerical_solution.y_star, 
             'ro', markersize=15, label=f'Optimal: $({numerical_solution.x_star:.2f}, {numerical_solution.y_star:.2f})$',
             zorder=5)
    
    # Formatting
    ax2d.set_xlabel('Good X', fontsize=14, fontweight='bold')
    ax2d.set_ylabel('Good Y', fontsize=14, fontweight='bold')
    ax2d.set_title('Utility Maximization: Indifference Curves and Budget Constraint', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2d.legend(fontsize=12, loc='upper right')
    ax2d.grid(True, alpha=0.3, linestyle='--')
    ax2d.set_xlim(0, _x_range[1])
    ax2d.set_ylim(0, _y_range[1])
    
    plt.tight_layout()
    
    # Clean up temporary variables
    del _x_range, _y_range, _x, _y, _X, _Y, _Z, _contour, _x_budget, _y_budget
    
    mo.md("## Step 5: Visualization - Indifference Curves")
    return ax2d, fig2d


@app.cell
def __(fig2d):
    """Cell 12: Display 2D Plot"""
    fig2d
    return


@app.cell
def __(go, np, numerical_solution, problem, utility_func):
    """Cell 13: 3D Interactive Visualization with Plotly"""
    
    # Create meshgrid for 3D plot
    _x3d = np.linspace(0.1, min(problem.income/problem.px * 1.2, 80), 60)
    _y3d = np.linspace(0.1, min(problem.income/problem.py * 1.2, 80), 60)
    _X3d, _Y3d = np.meshgrid(_x3d, _y3d)
    
    # Compute utility surface
    try:
        _Z3d = utility_func(_X3d, _Y3d)
    except:
        _Z3d = np.zeros_like(_X3d)
        for i in range(_X3d.shape[0]):
            for j in range(_X3d.shape[1]):
                try:
                    _Z3d[i,j] = utility_func(_X3d[i,j], _Y3d[i,j])
                except:
                    _Z3d[i,j] = np.nan
    
    # Create surface plot
    fig3d = go.Figure(data=[
        go.Surface(
            x=_X3d, y=_Y3d, z=_Z3d,
            colorscale='Viridis',
            opacity=0.8,
            name='Utility Surface'
        )
    ])
    
    # Add budget constraint plane
    _x_budget_3d = np.linspace(0, problem.income/problem.px, 30)
    _y_budget_3d = np.linspace(0, problem.income/problem.py, 30)
    _X_budget, _Y_budget = np.meshgrid(_x_budget_3d, _y_budget_3d)
    _Z_budget = np.where(
        (problem.px * _X_budget + problem.py * _Y_budget) <= problem.income,
        utility_func(_X_budget, _Y_budget),
        np.nan
    )
    
    fig3d.add_trace(go.Surface(
        x=_X_budget, y=_Y_budget, z=_Z_budget,
        colorscale='Reds',
        opacity=0.4,
        showscale=False,
        name='Feasible Set'
    ))
    
    # Add optimal point
    _z_opt = utility_func(numerical_solution.x_star, numerical_solution.y_star)
    fig3d.add_trace(go.Scatter3d(
        x=[numerical_solution.x_star],
        y=[numerical_solution.y_star],
        z=[_z_opt],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name=f'Optimal Point'
    ))
    
    # Add vertical line to optimal point
    fig3d.add_trace(go.Scatter3d(
        x=[numerical_solution.x_star, numerical_solution.x_star],
        y=[numerical_solution.y_star, numerical_solution.y_star],
        z=[0, _z_opt],
        mode='lines',
        line=dict(color='red', width=4, dash='dash'),
        name='Optimal Height'
    ))
    
    # Update layout
    fig3d.update_layout(
        title='Interactive 3D Utility Surface<br><sub>Rotate, zoom, and pan to explore</sub>',
        scene=dict(
            xaxis_title='Good X',
            yaxis_title='Good Y',
            zaxis_title='Utility U(x,y)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        width=1000,
        height=700,
        showlegend=True
    )
    
    # Clean up
    del _x3d, _y3d, _X3d, _Y3d, _Z3d, _x_budget_3d, _y_budget_3d, _X_budget, _Y_budget, _Z_budget, _z_opt
    
    mo.md("## Step 6: 3D Interactive Visualization")
    return (fig3d,)


@app.cell
def __(fig3d):
    """Cell 14: Display 3D Interactive Plot"""
    fig3d
    return


@app.cell
def __(mo):
    """Cell 15: Comparative Statics Controls"""
    
    comparative_statics_selector = mo.ui.dropdown(
        options={
            'Price of X (px)': 'px',
            'Price of Y (py)': 'py',
            'Income (I)': 'income',
            'Preference Parameter (α)': 'alpha'
        },
        value='px',
        label="Select parameter for comparative statics"
    )
    
    cs_start = mo.ui.number(start=0.1, stop=100, value=1, step=0.1, label="Start value")
    cs_end = mo.ui.number(start=0.1, stop=100, value=10, step=0.1, label="End value")
    cs_steps = mo.ui.slider(start=10, stop=100, value=30, label="Number of steps")
    
    cs_controls = mo.vstack([
        mo.md("## Step 7: Comparative Statics Analysis"),
        mo.md("Analyze how optimal choices change as parameters vary"),
        comparative_statics_selector,
        mo.hstack([cs_start, cs_end, cs_steps])
    ])
    
    cs_controls
    return (
        comparative_statics_selector,
        cs_controls,
        cs_end,
        cs_start,
        cs_steps,
    )


@app.cell
def __(
    NonlinearConstraint,
    UtilityMaxProblem,
    comparative_statics_selector,
    cs_end,
    cs_start,
    cs_steps,
    function_selector,
    minimize,
    np,
    pd,
    problem,
):
    """Cell 16: Perform Comparative Statics"""
    
    # Get parameter range
    _param_values = np.linspace(cs_start.value, cs_end.value, cs_steps.value)
    _param_name = comparative_statics_selector.value
    
    # Storage for results
    _cs_results = []
    
    # Run optimization for each parameter value
    for _val in _param_values:
        # Create modified problem
        if _param_name == 'px':
            _temp_problem = UtilityMaxProblem(
                problem.utility_func, _val, problem.py, problem.income,
                problem.alpha, problem.rho
            )
        elif _param_name == 'py':
            _temp_problem = UtilityMaxProblem(
                problem.utility_func, problem.px, _val, problem.income,
                problem.alpha, problem.rho
            )
        elif _param_name == 'income':
            _temp_problem = UtilityMaxProblem(
                problem.utility_func, problem.px, problem.py, _val,
                problem.alpha, problem.rho
            )
        else:  # alpha
            if function_selector.value == 'cobb_douglas':
                _temp_utility = lambda x, y: x**_val * y**(1-_val)
                _temp_problem = UtilityMaxProblem(
                    _temp_utility, problem.px, problem.py, problem.income,
                    _val, problem.rho
                )
            else:
                continue  # Skip if alpha not applicable
        
        # Solve
        _obj = lambda vars: -_temp_problem.utility_func(vars[0], vars[1]) if vars[0] > 0 and vars[1] > 0 else 1e10
        _cons = NonlinearConstraint(
            lambda vars: _temp_problem.px * vars[0] + _temp_problem.py * vars[1],
            _temp_problem.income, _temp_problem.income
        )
        _result = minimize(
            _obj, [_temp_problem.income/(2*_temp_problem.px), _temp_problem.income/(2*_temp_problem.py)],
            method='SLSQP', constraints=_cons,
            bounds=[(1e-6, _temp_problem.income/_temp_problem.px), (1e-6, _temp_problem.income/_temp_problem.py)]
        )
        
        if _result.success:
            _cs_results.append({
                'parameter': _val,
                'x_star': _result.x[0],
                'y_star': _result.x[1],
                'utility': -_result.fun
            })
    
    cs_results_df = pd.DataFrame(_cs_results)
    
    # Clean up
    del _param_values, _param_name, _cs_results, _val, _temp_problem, _obj, _cons, _result
    return (cs_results_df,)


@app.cell
def __(comparative_statics_selector, cs_results_df, mo, plt):
    """Cell 17: Plot Comparative Statics Results"""
    
    fig_cs, axes_cs = plt.subplots(1, 3, figsize=(16, 5))
    
    _param_label = comparative_statics_selector.label.split('(')[1].rstrip(')')
    
    # Plot x* vs parameter
    axes_cs[0].plot(cs_results_df['parameter'], cs_results_df['x_star'], 'b-', linewidth=2.5)
    axes_cs[0].set_xlabel(f'{_param_label}', fontsize=12)
    axes_cs[0].set_ylabel('Optimal $x^*$', fontsize=12)
    axes_cs[0].set_title('Demand for Good X', fontsize=13, fontweight='bold')
    axes_cs[0].grid(True, alpha=0.3)
    
    # Plot y* vs parameter
    axes_cs[1].plot(cs_results_df['parameter'], cs_results_df['y_star'], 'g-', linewidth=2.5)
    axes_cs[1].set_xlabel(f'{_param_label}', fontsize=12)
    axes_cs[1].set_ylabel('Optimal $y^*$', fontsize=12)
    axes_cs[1].set_title('Demand for Good Y', fontsize=13, fontweight='bold')
    axes_cs[1].grid(True, alpha=0.3)
    
    # Plot utility vs parameter
    axes_cs[2].plot(cs_results_df['parameter'], cs_results_df['utility'], 'r-', linewidth=2.5)
    axes_cs[2].set_xlabel(f'{_param_label}', fontsize=12)
    axes_cs[2].set_ylabel('Maximum Utility $U^*$', fontsize=12)
    axes_cs[2].set_title('Indirect Utility Function', fontsize=13, fontweight='bold')
    axes_cs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    del _param_label
    
    mo.md("### Comparative Statics Results")
    return axes_cs, fig_cs


@app.cell
def __(fig_cs):
    """Cell 18: Display Comparative Statics Plots"""
    fig_cs
    return


@app.cell
def __(cs_results_df, mo):
    """Cell 19: Display Comparative Statics Table"""
    
    mo.ui.table(
        cs_results_df.round(4),
        label="Comparative Statics Data",
        selection=None
    )
    return


@app.cell
def __(mo):
    """Cell 20: Economic Insights and Interpretation"""
    
    mo.md("""
    ## Economic Insights
    
    ### Key Concepts Demonstrated
    
    1. **Tangency Condition**: At the optimum, the marginal rate of substitution (MRS) equals the price ratio:
       $$MRS_{xy} = \\frac{MU_x}{MU_y} = \\frac{p_x}{p_y}$$
    
    2. **Budget Exhaustion**: For monotonic preferences, the optimal bundle lies on (not below) the budget line.
    
    3. **Comparative Statics**: How optimal choices respond to parameter changes:
       - **Income effects**: Normal goods have positive income derivatives
       - **Price effects**: Combining substitution and income effects
       - **Cross-price effects**: How demand for one good changes with another's price
    
    4. **Properties of Demand Functions**:
       - Homogeneity of degree zero in prices and income
       - Slutsky symmetry
       - Negative own-price effects (substitution dominates)
    
    ### Extensions to Explore
    
    - Slutsky decomposition (income vs substitution effects)
    - Expenditure minimization (dual problem)
    - Revealed preference and WARP
    - Expected utility under uncertainty
    
    ---
    
    *Created with marimo - reactive Python notebooks*
    """)
    return


@app.cell
def __(mo):
    """Cell 21: Export and Save Functionality"""
    
    export_button = mo.ui.button(label="📥 Export Results to CSV", on_click=lambda: None)
    save_figure_button = mo.ui.button(label="💾 Save Figures", on_click=lambda: None)
    
    mo.hstack([
        export_button,
        save_figure_button,
        mo.md("*Coming soon: automated exports*")
    ])
    return export_button, save_figure_button


if __name__ == "__main__":
    app.run()