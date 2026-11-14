"""Calcula la forma de una viga usando sus extremos y puntos intermedios."""

import numpy as np
import deffuncionespvf as dfp


def diffincentr(
    t, y_0, y_n, n, h, right_bc="dirichlet", kappa=0.0,
    c_func=None, forcing_func=None
):
    """Devuelve la deflexión aproximada de la viga en cada punto de la malla."""
    # Comprueba que los puntos y el paso formen una malla regular válida.
    t = np.asarray(t, dtype=float)
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n debe ser un entero mayor o igual que 2.")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h debe ser un número positivo y finito.")
    if right_bc not in ("dirichlet", "robin"):
        raise ValueError("right_bc debe ser 'dirichlet' o 'robin'.")
    if right_bc == "robin" and not np.isfinite(kappa):
        raise ValueError("kappa debe ser un número finito.")
    c_func = dfp.C if c_func is None else c_func
    forcing_func = dfp.f if forcing_func is None else forcing_func
    if not callable(c_func) or not callable(forcing_func):
        raise ValueError("c_func y forcing_func deben ser funciones.")
    if t.ndim != 1 or t.size != n + 1:
        raise ValueError("t debe contener exactamente n + 1 puntos.")
    if not np.all(np.isfinite(t)) or not np.allclose(
        np.diff(t), h, rtol=1e-10, atol=1e-12
    ):
        raise ValueError("Los puntos de t deben estar uniformemente espaciados por h.")

    # Reserva la respuesta y fija el extremo izquierdo conocido.
    y = np.zeros(n + 1)
    y[0] = y_0
    if right_bc == "dirichlet":
        y[-1] = y_n
        system_size = n - 1
    else:
        system_size = n

    # Prepara las ecuaciones que conectan cada punto con sus vecinos.
    mat = np.zeros((system_size, system_size))
    vec = np.zeros(system_size)

    # Calcula las filas de los puntos interiores de la viga.
    for i in range(n - 1):
        x_i = t[i + 1]
        lower = dfp.A(x_i) / h**2 - dfp.B(x_i) / (2 * h)
        diagonal = -2 * dfp.A(x_i) / h**2 + c_func(x_i)
        upper = dfp.A(x_i) / h**2 + dfp.B(x_i) / (2 * h)

        mat[i, i] = diagonal
        vec[i] = forcing_func(x_i)
        if i == 0:
            vec[i] -= lower * y_0
        else:
            mat[i, i - 1] = lower
        if right_bc == "dirichlet" and i == n - 2:
            vec[i] -= upper * y_n
        else:
            mat[i, i + 1] = upper

    if right_bc == "robin":
        # Usa el apoyo elástico para escribir la última ecuación sin un punto extra.
        x_n = t[-1]
        mat[-1, -2] = 2 * dfp.A(x_n)
        mat[-1, -1] = (
            -(2 + 2 * h * kappa) * dfp.A(x_n)
            - h**2 * dfp.B(x_n) * kappa
            + h**2 * c_func(x_n)
        )
        vec[-1] = h**2 * forcing_func(x_n)
        # Encuentra todos los valores desconocidos y los coloca en la respuesta.
        solution = np.linalg.solve(mat, vec)
        y[1:] = solution
    else:
        y[1:-1] = np.linalg.solve(mat, vec)

    return y