"""
Módulo en el que se definen las funciones que se utilizan
en los métodos de solución de PVIs.
* funced: función presente en la ecuación diferencial.
* funcsolex: solución exacta del PVI en caso de existir.
* funsolref: solución de referencia del PVI utilizando solve_ivp.
"""

import numpy as np
from scipy.integrate import solve_ivp

# Parámetros del modelo de vibración de la pasarela (Equipo n = 2)
MASS = 20.0
STIFFNESS = 20.0
INITIAL_DISPLACEMENT = 1.0
INITIAL_VELOCITY = 0.0
T_END = 15.0
FORCE_AMPLITUDE = 7.0
QUADRATIC_DAMPING = 4.0
LINEAR_DAMPING_VALUES = (5.0, 40.0, 200.0)
FREQUENCIES = np.linspace(0.2, 3.0, 29)


def bridge_linear(t, y, damping, force_amplitude=0.0, omega=0.0):
    """Sistema [x', v'] de la pasarela con amortiguamiento lineal."""
    state = np.asarray(y, dtype=float)
    if state.shape != (2,):
        raise ValueError("El estado de la pasarela debe contener [x, v].")
    force = force_amplitude * np.cos(omega * t)
    acceleration = (
        force - damping * state[1] - STIFFNESS * state[0]
    ) / MASS
    return np.array([state[1], acceleration])


def bridge_quadratic(t, y, force_amplitude=0.0, omega=0.0, linear_damping=0.0):
    """Sistema [x', v'] con amortiguamiento lineal y cuadrático."""
    state = np.asarray(y, dtype=float)
    if state.shape != (2,):
        raise ValueError("El estado de la pasarela debe contener [x, v].")
    force = force_amplitude * np.cos(omega * t)
    acceleration = (
        force
        - linear_damping * state[1]
        - QUADRATIC_DAMPING * state[1] * abs(state[1])
        - STIFFNESS * state[0]
    ) / MASS
    return np.array([state[1], acceleration])


def bridge_nonlinear_reference(
    t0, t_end, y0, force_amplitude, omega, t_eval=None, linear_damping=0.0
):
    """Referencia RK45 de SciPy para la Variante C."""
    state = np.asarray(y0, dtype=float)
    if state.shape != (2,):
        raise ValueError("y0 debe contener [x0, v0].")
    if not np.isfinite(t0) or not np.isfinite(t_end) or t_end <= t0:
        raise ValueError("Se requiere un intervalo temporal finito y creciente.")
    if t_eval is not None:
        t_eval = np.asarray(t_eval, dtype=float)
        if t_eval.ndim != 1 or np.any(np.diff(t_eval) <= 0):
            raise ValueError("t_eval debe ser un vector estrictamente creciente.")
        if t_eval[0] < t0 or t_eval[-1] > t_end:
            raise ValueError("Los valores de t_eval deben estar dentro de [t0, t_end].")

    solution = solve_ivp(
        lambda t, y: bridge_quadratic(
            t, y, force_amplitude, omega, linear_damping
        ),
        (t0, t_end),
        state,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(f"solve_ivp no convergió: {solution.message}")
    return solution


# Función f de la EDO (y'' = g(t,y,y')) 
def funced(t,y):
    yder2 = 4*y[0] + 12*t        
    return yder2

# Derivada de funcraiz
def funcsolex(t):
    y = 3*np.exp(2*t) + np.exp(-2*t) - 3*t
    return y

# Solución de referencia del PVI
def funcsolref(t_0, T, y_0):
    sol = solve_ivp(lambda t, y: [y[1], 4*y[0] + 12*t], [t_0, T], y_0, rtol = 1e-10)
    return sol