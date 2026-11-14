"""Validación y comparación de los tres modelos de deflexión de la viga."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import deffuncionespvf as dfp
from metdf_solpvf import diffincentr


def exercise_1f():
    """Valida la condición Robin con y'' + 3y = 6x."""
    original_coefficients = dfp.A, dfp.B, dfp.C, dfp.f
    dfp.A = lambda x: np.ones_like(np.asarray(x, dtype=float))
    dfp.B = lambda x: np.zeros_like(np.asarray(x, dtype=float))
    dfp.C = lambda x: 3.0 * np.ones_like(np.asarray(x, dtype=float))
    dfp.f = lambda x: 6.0 * np.asarray(x, dtype=float)

    try:
        h = 0.01
        x = np.linspace(0.0, 1.0, int(round(1.0 / h)) + 1)
        y = diffincentr(x, 0.0, None, len(x) - 1, h, right_bc="robin", kappa=1.0)
        root = np.sqrt(3.0)
        coefficient = -4.0 / (np.sin(root) + root * np.cos(root))
        exact = coefficient * np.sin(root * x) + 2.0 * x
        rms = float(np.sqrt(np.mean((y - exact) ** 2)))
        ghost = y[-2] - 2.0 * h * y[-1]
        robin_residual = (ghost - y[-2]) / (2.0 * h) + y[-1]
        print("Validación ejercicio 1f: y'' + 3y = 6x, y(0)=0, y(1)+y'(1)=0")
        print(f"  Error RMS frente a la solución exacta: {rms:.10e}")
        print(f"  Residuo Robin centrado (nodo ficticio): {robin_residual:.10e}")
        if rms > 3e-4 or abs(robin_residual) > 1e-12:
            raise AssertionError("Falló la validación de la condición Robin.")
    finally:
        dfp.A, dfp.B, dfp.C, dfp.f = original_coefficients


def main():
    exercise_1f()

    h = 6.0
    n = int(round(dfp.L / h))
    h = dfp.L / n
    x = np.linspace(0.0, dfp.L, n + 1)

    base = dfp.funcsolex(x)
    variant_a = diffincentr(x, 0.0, 0.0, n, h)

    reference = dfp.funcsolref(x, np.zeros((2, x.size)), 0.0, 0.0)
    variant_a_reference = reference.sol(x)[0]
    rms_a_reference = float(np.sqrt(np.mean((variant_a - variant_a_reference) ** 2)))
    rms_base_a = float(np.sqrt(np.mean((base - variant_a) ** 2)))

    variant_b = diffincentr(
        x, 0.0, None, n, h, right_bc="robin", kappa=dfp.KAPPA,
        c_func=dfp.C_const, forcing_func=dfp.f_const
    )
    center_index = n // 2
    if not np.isclose(variant_b[center_index], 3.845e-3, rtol=2e-3):
        raise AssertionError("La deflexión central de la Variante B no coincide.")
    if not np.isclose(variant_b[-1], 7.225e-4, rtol=2e-3):
        raise AssertionError("La deflexión en x=l de la Variante B no coincide.")

    print("\nComparación de deflexiones (unidades: pulg; malla cada 6 pulg)")
    print("| x (pulg) | Caso base | Variante A | Variante B |")
    print("|---:|---:|---:|---:|")
    for x_i, base_i, a_i, b_i in zip(x, base, variant_a, variant_b):
        print(f"| {x_i:.1f} | {base_i:.10e} | {a_i:.10e} | {b_i:.10e} |")
    print(f"\nError RMS Variante A frente a solve_bvp: {rms_a_reference:.10e} pulg")
    print(f"Error RMS Caso base frente a Variante A: {rms_base_a:.10e} pulg")
    print(
        "Variante B (I=I0, apoyo Robin): "
        f"w(72) = {variant_b[center_index]:.10e} pulg, "
        f"w(144) = {variant_b[-1]:.10e} pulg."
    )

    for label, values in (
        ("Caso base", base),
        ("Variante A", variant_a),
        ("Variante B", variant_b),
    ):
        index = int(np.argmax(np.abs(values)))
        print(
            f"{label}: |w_max| = {abs(values[index]):.10e} pulg, "
            f"x_max = {x[index]:.1f} pulg (w = {values[index]:.10e} pulg)."
        )

    dense_x = np.linspace(0.0, dfp.L, 500)
    plt.figure(figsize=(10, 6))
    plt.plot(dense_x, dfp.funcsolex(dense_x), label="Caso base (analítico)")
    plt.plot(x, variant_a, "o-", label="Variante A (I variable, diferencias finitas)")
    plt.plot(
        x, variant_b, "s-",
        label="Variante B (I constante, apoyo Robin, diferencias finitas)"
    )
    plt.xlabel("Posición x (pulg)")
    plt.ylabel("Deflexión w(x) (pulg)")
    plt.title("Deflexión estática de la viga — Equipo n = 2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    figure_path = Path(__file__).with_name("comparacion_fase1.png")
    plt.savefig(figure_path, dpi=180)
    print(f"\nGráfica guardada en: {figure_path}")
    plt.show()


if __name__ == "__main__":
    main()
