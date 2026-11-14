"""Método lineal del disparo para problemas de frontera de segundo orden."""

import numpy as np

from met_solpvi_sist import rk4_meth


def disparo_lineal(
    f_pvi, x0, xf, y0_val, yf_val, h, z1=0.0, z2=1.0, return_slope=False
):
    """Ajusta la pendiente inicial hasta que la curva llegue al extremo deseado."""
    # Revisa los datos antes de empezar los intentos.
    if not callable(f_pvi):
        raise TypeError("f_pvi debe ser una función callable.")
    if not all(np.isfinite(value) for value in (x0, xf, y0_val, yf_val, h, z1, z2)):
        raise ValueError("Los datos del disparo deben ser finitos.")
    if xf <= x0:
        raise ValueError("Se requiere xf > x0.")
    if h <= 0:
        raise ValueError("h debe ser positivo.")
    if z1 == z2:
        raise ValueError("z1 y z2 deben ser diferentes.")

    nit = int(round((xf - x0) / h))
    if nit < 1:
        raise ValueError("El intervalo y el paso deben producir al menos un paso.")
    step = (xf - x0) / nit

    # Prueba una pendiente y mide cuánto se aparta del valor final pedido.
    def integrate(initial_slope):
        x, states = rk4_meth(
            f_pvi, x0, [y0_val, initial_slope], nit, step
        )
        return x, states, states[-1, 0] - yf_val

    # Usa dos pendientes iniciales para calcular una mejor aproximación.
    x, _, phi1 = integrate(z1)
    _, _, phi2 = integrate(z2)
    denominator = phi2 - phi1
    scale = max(1.0, abs(phi1), abs(phi2))
    if abs(denominator) <= np.finfo(float).eps * scale:
        raise ValueError("Los disparos no definen una secante para calcular z_star.")

    # Calcula la pendiente que debería alcanzar el extremo con más precisión.
    z_star = z1 - phi1 * (z2 - z1) / denominator
    # Repite el recorrido con la pendiente final y devuelve la curva.
    x, states, _ = integrate(z_star)
    result = (x, states[:, 0])
    if return_slope:
        return result + (z_star,)
    return result
