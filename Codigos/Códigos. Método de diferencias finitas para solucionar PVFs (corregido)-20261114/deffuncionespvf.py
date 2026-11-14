"""Guarda los datos y las fórmulas usadas para calcular la deflexión de la viga."""

import numpy as np


# Medidas y propiedades de la viga del equipo.
L = 144.0
Q = 140.0 / 12.0
E = 3.0e7
S = 1000.0
I0 = 625.0
GAMMA = 0.4
KAPPA = 0.10


def A(x):
    """Devuelve cuánto influye la curvatura de la viga en el modelo."""
    return np.ones_like(np.asarray(x, dtype=float))


def B(x):
    """Devuelve el aporte asociado al cambio de pendiente."""
    return np.zeros_like(np.asarray(x, dtype=float))


def C(x):
    """Calcula el aporte de la deflexión cuando la rigidez cambia con la posición."""
    inertia = I0 * (1.0 + GAMMA * np.asarray(x, dtype=float) / L)
    return -S / (E * inertia)


def C_const(x):
    """Calcula el aporte de la deflexión cuando la rigidez se mantiene fija."""
    x = np.asarray(x, dtype=float)
    return np.full_like(x, -S / (E * I0))


def f(x):
    """Calcula el efecto de la carga cuando la rigidez cambia con la posición."""
    x = np.asarray(x, dtype=float)
    inertia = I0 * (1.0 + GAMMA * x / L)
    return Q * x * (x - L) / (2.0 * E * inertia)


def f_const(x):
    """Calcula el efecto de la carga cuando la rigidez se mantiene fija."""
    x = np.asarray(x, dtype=float)
    return Q * x * (x - L) / (2.0 * E * I0)


def funcsolex(x):
    """Devuelve la deflexión exacta del caso base en las posiciones indicadas."""
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
    """Calcula una curva de referencia para la viga con rigidez variable."""
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

    # Describe cómo cambian la posición y la pendiente a lo largo de la viga.
    def ode(x_values, y_values):
        return np.vstack(
            (
                y_values[1],
                -(B(x_values) * y_values[1] + C(x_values) * y_values[0]
                  - f(x_values)) / A(x_values),
            )
        )

    # Obliga a que la viga cumpla las deflexiones indicadas en ambos extremos.
    def boundary_conditions(ya, yb):
        return np.array([ya[0] - y_0, yb[0] - y_n])

    solution = solve_bvp(ode, boundary_conditions, x, ysr)
    if not solution.success:
        raise RuntimeError(f"solve_bvp no convergió: {solution.message}")
    return solution
