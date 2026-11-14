"""Coeficientes y soluciones de referencia del modelo de deflexión de la viga."""

import numpy as np


L = 144.0
Q = 140.0 / 12.0
E = 3.0e7
S = 1000.0
I0 = 625.0
GAMMA = 0.4
KAPPA = 0.10


def A(x):
    """Coeficiente de w'' para la viga."""
    return np.ones_like(np.asarray(x, dtype=float))


def B(x):
    """Coeficiente de w' para la viga."""
    return np.zeros_like(np.asarray(x, dtype=float))


def C(x):
    """Coeficiente de w para la inercia variable."""
    inertia = I0 * (1.0 + GAMMA * np.asarray(x, dtype=float) / L)
    return -S / (E * inertia)


def C_const(x):
    """Coeficiente de w para la inercia constante I0."""
    x = np.asarray(x, dtype=float)
    return np.full_like(x, -S / (E * I0))


def f(x):
    """Carga distribuida dividida por E I(x)."""
    x = np.asarray(x, dtype=float)
    inertia = I0 * (1.0 + GAMMA * x / L)
    return Q * x * (x - L) / (2.0 * E * inertia)


def f_const(x):
    """Carga distribuida dividida por E I0 para el caso base."""
    x = np.asarray(x, dtype=float)
    return Q * x * (x - L) / (2.0 * E * I0)


def funcsolex(x):
    """Solución exacta del caso base con coeficientes constantes."""
    x = np.asarray(x, dtype=float)
    lambda_val = np.sqrt(S / (E * I0))
    scale = Q * E * I0 / S**2
    half_length = lambda_val * L / 2.0
    homogeneous_difference = (
        2.0
        * np.sinh(lambda_val * x / 2.0)
        * np.sinh(lambda_val * (x - L) / 2.0)
        / np.cosh(half_length)
    )
    particular_scaled = lambda_val**2 * x * (L - x) / 2.0
    return scale * (homogeneous_difference + particular_scaled)


def funcsolref(x, ysr=None, y_0=0.0, y_n=0.0):
    """Solución de referencia de la Variante A mediante scipy.solve_bvp."""
    from scipy.integrate import solve_bvp

    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("x debe ser una malla unidimensional estrictamente creciente.")
    if ysr is None:
        ysr = np.zeros((2, x.size), dtype=float)
    else:
        ysr = np.asarray(ysr, dtype=float)
        if ysr.shape != (2, x.size):
            raise ValueError("ysr debe tener forma (2, len(x)).")

    def ode(x_values, y_values):
        return np.vstack(
            (
                y_values[1],
                -(B(x_values) * y_values[1] + C(x_values) * y_values[0]
                  - f(x_values)) / A(x_values),
            )
        )

    def boundary_conditions(ya, yb):
        return np.array([ya[0] - y_0, yb[0] - y_n])

    solution = solve_bvp(ode, boundary_conditions, x, ysr)
    if not solution.success:
        raise RuntimeError(f"solve_bvp no convergió: {solution.message}")
    return solution
