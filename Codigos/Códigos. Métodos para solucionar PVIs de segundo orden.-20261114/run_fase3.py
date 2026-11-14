"""Compara diferencias finitas y método del disparo para la viga."""

from pathlib import Path
import importlib.util
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
from met_disparo import disparo_lineal

PVF_DIRECTORY = (
    Path(__file__).parent.parent
    / "Códigos. Método de diferencias finitas para solucionar PVFs (corregido)-20261114"
)


def load_pvf_module(module_name):
    module_path = PVF_DIRECTORY / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el módulo {module_path}.")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


beam: ModuleType = load_pvf_module("deffuncionespvf")
finite_difference_module = load_pvf_module("metdf_solpvf")
diffincentr = finite_difference_module.diffincentr


STEP_SIZE = 6.0
OUTPUT_PATH = Path(__file__).with_name("fase3_disparo.png")


def beam_rhs(inertia):
    def funced(x, state):
        force_term = beam.Q * x * (x - beam.L) / 2.0
        curvature = (
            beam.S * state[0] + force_term
        ) / (beam.E * inertia(x))
        return np.array([state[1], curvature])

    return funced


def finite_difference_base(x, intervals):
    original = beam.A, beam.B, beam.C, beam.f
    beam.A = lambda value: np.ones_like(np.asarray(value, dtype=float))
    beam.B = lambda value: np.zeros_like(np.asarray(value, dtype=float))
    beam.C = lambda value: np.full_like(
        np.asarray(value, dtype=float), -beam.S / (beam.E * beam.I0)
    )
    beam.f = lambda value: (
        beam.Q * np.asarray(value, dtype=float)
        * (np.asarray(value, dtype=float) - beam.L)
        / (2.0 * beam.E * beam.I0)
    )
    try:
        return diffincentr(x, 0.0, 0.0, intervals, STEP_SIZE)
    finally:
        beam.A, beam.B, beam.C, beam.f = original


def rms(first, second):
    return float(np.sqrt(np.mean((np.asarray(first) - np.asarray(second)) ** 2)))


def main():
    intervals = int(round(beam.L / STEP_SIZE))
    h = beam.L / intervals
    x = np.linspace(0.0, beam.L, intervals + 1)

    exact_base = beam.funcsolex(x)
    fd_base = finite_difference_base(x, intervals)
    shot_x_base, shot_base, slope_base = disparo_lineal(
        beam_rhs(
            lambda position: beam.I0
            * np.ones_like(np.asarray(position, dtype=float))
        ),
        0.0, beam.L, 0.0, 0.0, h, return_slope=True
    )

    reference_a = beam.funcsolref(
        x, np.zeros((2, x.size)), y_0=0.0, y_n=0.0
    )
    reference_a_values = reference_a.sol(x)[0]
    fd_a = diffincentr(x, 0.0, 0.0, intervals, h)
    shot_x_a, shot_a, slope_a = disparo_lineal(
        beam_rhs(lambda position: beam.I0 * (1.0 + beam.GAMMA * position / beam.L)),
        0.0, beam.L, 0.0, 0.0, h, return_slope=True
    )

    if not np.allclose(x, shot_x_base) or not np.allclose(x, shot_x_a):
        raise AssertionError("Las mallas de diferencias finitas y disparo no coinciden.")

    rms_base = rms(fd_base, shot_base)
    rms_a = rms(fd_a, shot_a)
    if max(rms_base, rms_a) > 1e-4:
        raise AssertionError(
            f"Disparo y diferencias finitas no concuerdan: "
            f"RMS base={rms_base:.3e}, Variante A={rms_a:.3e}."
        )

    print(f"Malla: h={h:g} pulg, {intervals} subintervalos")
    print(f"Pendiente inicial Caso Base z* = {slope_base:.12e} pulg/pulg")
    print(f"Pendiente inicial Variante A z* = {slope_a:.12e} pulg/pulg")
    print(f"Error RMS (Disparo vs DF), Caso Base: {rms_base:.10e} pulg")
    print(f"Error RMS (Disparo vs DF), Variante A: {rms_a:.10e} pulg")
    print("\n| x (pulg) | Base exacta | Base DF | Base disparo | A solve_bvp | A DF | A disparo |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for row in zip(x, exact_base, fd_base, shot_base,
                   reference_a_values, fd_a, shot_a):
        print(
            f"| {row[0]:.1f} | {row[1]:.10e} | {row[2]:.10e} | "
            f"{row[3]:.10e} | {row[4]:.10e} | {row[5]:.10e} | "
            f"{row[6]:.10e} |"
        )

    dense_x = np.linspace(0.0, beam.L, 500)
    plt.figure(figsize=(11, 6.5))
    plt.plot(dense_x, beam.funcsolex(dense_x), "k--", label="Caso base: exacta")
    plt.plot(
        dense_x, reference_a.sol(dense_x)[0], color="0.45", linestyle=":",
        label="Variante A: solve_bvp"
    )
    plt.plot(x, fd_base, "o-", label="Caso base: diferencias finitas")
    plt.plot(x, shot_base, "s--", label="Caso base: disparo RK4")
    plt.plot(x, fd_a, "^-", label="Variante A: diferencias finitas")
    plt.plot(x, shot_a, "d--", label="Variante A: disparo RK4")
    plt.xlabel("Posición x (pulg)")
    plt.ylabel("Deflexión w(x) (pulg)")
    plt.title("Deflexión de la viga: diferencias finitas y método del disparo")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=180)
    print(f"\nGráfica guardada: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
