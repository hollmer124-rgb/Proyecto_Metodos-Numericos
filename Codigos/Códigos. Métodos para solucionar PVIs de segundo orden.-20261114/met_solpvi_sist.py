"""Ofrece cuatro maneras de avanzar paso a paso una o varias cantidades."""

import numpy as np


def _integrate(funced, t0, y0, nit, h, method):
    """Prepara los resultados y repite el avance elegido en cada paso."""
    # Comprueba que la función, el paso y las condiciones iniciales sean válidos.
    if not callable(funced):
        raise TypeError("funced debe ser una función callable.")
    if not isinstance(nit, (int, np.integer)) or nit < 0:
        raise ValueError("nit debe ser un entero no negativo.")
    if not np.isfinite(t0):
        raise ValueError("t0 debe ser finito.")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h debe ser positivo y finito.")

    # Convierte el inicio en una lista de valores que el programa pueda calcular.
    state = np.asarray(y0, dtype=float)
    if state.ndim == 0:
        state = state.reshape(1)
    if state.ndim != 1 or state.size == 0:
        raise ValueError("y0 debe ser un escalar o un vector unidimensional no vacío.")

    # Verifica que la función entregue un cambio por cada valor inicial.
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

    # Guarda todos los tiempos y resultados, incluido el punto inicial.
    times = t0 + h * np.arange(nit + 1, dtype=float)
    states = np.empty((nit + 1, state.size), dtype=float)
    states[0] = state
    for i in range(nit):
        states[i + 1] = method(evaluate, times[i], states[i], h)
    return times, states


def eul_meth(funced, t0, y0, nit, h):
    """Avanza usando solo la información disponible al inicio de cada paso."""
    def step(fun, t, y, step_size):
        return y + step_size * fun(t, y)

    return _integrate(funced, t0, y0, nit, h, step)


def heun_meth(funced, t0, y0, nit, h):
    """Mejora cada paso promediando los cambios al inicio y al final."""
    def step(fun, t, y, step_size):
        k1 = fun(t, y)
        k2 = fun(t + step_size, y + step_size * k1)
        return y + step_size * (k1 + k2) / 2.0

    return _integrate(funced, t0, y0, nit, h, step)


def pm_meth(funced, t0, y0, nit, h):
    """Calcula el cambio en la mitad del paso y lo usa para avanzar."""
    def step(fun, t, y, step_size):
        k1 = fun(t, y)
        k2 = fun(t + step_size / 2.0, y + step_size * k1 / 2.0)
        return y + step_size * k2

    return _integrate(funced, t0, y0, nit, h, step)


def rk4_meth(funced, t0, y0, nit, h):
    """Combina cuatro estimaciones para obtener un avance más preciso."""
    def step(fun, t, y, step_size):
        k1 = fun(t, y)
        k2 = fun(t + step_size / 2.0, y + step_size * k1 / 2.0)
        k3 = fun(t + step_size / 2.0, y + step_size * k2 / 2.0)
        k4 = fun(t + step_size, y + step_size * k3)
        return y + step_size * (k1 + 2*k2 + 2*k3 + k4) / 6.0

    return _integrate(funced, t0, y0, nit, h, step)
