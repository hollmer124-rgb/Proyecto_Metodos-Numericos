"""Integradores explícitos para sistemas de EDOs de primer orden."""

import numpy as np


def _integrate(funced, t0, y0, nit, h, method):
    if not callable(funced):
        raise TypeError("funced debe ser una función callable.")
    if not isinstance(nit, (int, np.integer)) or nit < 0:
        raise ValueError("nit debe ser un entero no negativo.")
    if not np.isfinite(t0):
        raise ValueError("t0 debe ser finito.")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h debe ser positivo y finito.")

    state = np.asarray(y0, dtype=float)
    if state.ndim == 0:
        state = state.reshape(1)
    if state.ndim != 1 or state.size == 0:
        raise ValueError("y0 debe ser un escalar o un vector unidimensional no vacío.")

    def evaluate(t, y):
        derivative = np.asarray(funced(t, y.copy()), dtype=float)
        if derivative.ndim == 0 and y.size == 1:
            derivative = derivative.reshape(1)
        if derivative.shape != y.shape:
            raise ValueError(
                f"funced debe retornar un vector de forma {y.shape}; "
                f"se recibió {derivative.shape}."
            )
        return derivative

    times = t0 + h * np.arange(nit + 1, dtype=float)
    states = np.empty((nit + 1, state.size), dtype=float)
    states[0] = state
    for i in range(nit):
        states[i + 1] = method(evaluate, times[i], states[i], h)
    return times, states


def eul_meth(funced, t0, y0, nit, h):
    """Euler explícito; retorna tiempos y estados de forma (nit+1, N)."""
    def step(fun, t, y, step_size):
        return y + step_size * fun(t, y)

    return _integrate(funced, t0, y0, nit, h, step)


def heun_meth(funced, t0, y0, nit, h):
    """Heun explícito de segundo orden para un sistema vectorial."""
    def step(fun, t, y, step_size):
        k1 = fun(t, y)
        k2 = fun(t + step_size, y + step_size * k1)
        return y + step_size * (k1 + k2) / 2.0

    return _integrate(funced, t0, y0, nit, h, step)


def pm_meth(funced, t0, y0, nit, h):
    """Método explícito del punto medio de segundo orden."""
    def step(fun, t, y, step_size):
        k1 = fun(t, y)
        k2 = fun(t + step_size / 2.0, y + step_size * k1 / 2.0)
        return y + step_size * k2

    return _integrate(funced, t0, y0, nit, h, step)


def rk4_meth(funced, t0, y0, nit, h):
    """Runge-Kutta clásico de cuarto orden para un sistema vectorial."""
    def step(fun, t, y, step_size):
        k1 = fun(t, y)
        k2 = fun(t + step_size / 2.0, y + step_size * k1 / 2.0)
        k3 = fun(t + step_size / 2.0, y + step_size * k2 / 2.0)
        k4 = fun(t + step_size, y + step_size * k3)
        return y + step_size * (k1 + 2*k2 + 2*k3 + k4) / 6.0

    return _integrate(funced, t0, y0, nit, h, step)
