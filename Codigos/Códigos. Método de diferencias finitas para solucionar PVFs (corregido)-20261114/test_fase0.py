"""Validación de convergencia para el ejercicio 1b del taller de PVF."""

import numpy as np

import deffuncionespvf as dfp
from metdf_solpvf import diffincentr


def exact_solution(x):
    """Da el valor conocido de la curva para comparar el cálculo."""
    return x**4 - x**2 + 3


def test_single_interior_node():
    """Comprueba que el cálculo incluya los dos extremos cuando hay un punto interior."""
    x = np.linspace(1.0, 2.0, 3)
    h = 0.5
    x_i = x[1]
    lower = x_i**2/h**2 - (-5*x_i)/(2*h)
    diagonal = -2*x_i**2/h**2 + 8
    upper = x_i**2/h**2 + (-5*x_i)/(2*h)
    expected = (24 - lower*3 - upper*15)/diagonal
    actual = diffincentr(x, 3.0, 15.0, 2, h)
    if not np.isclose(actual[1], expected):
        raise AssertionError("El sistema no incorporó ambas condiciones de frontera.")


def main():
    # Configura la ecuación del ejercicio del taller.
    dfp.A = lambda x: np.asarray(x)**2
    dfp.B = lambda x: -5*np.asarray(x)
    dfp.C = lambda x: np.full_like(np.asarray(x), 8, dtype=float)
    dfp.f = lambda x: np.full_like(np.asarray(x), 24, dtype=float)

    # Resuelve el ejercicio con pasos cada vez más pequeños.
    test_single_interior_node()
    x_0, x_n = 1.0, 2.0
    y_0, y_n = exact_solution(x_0), exact_solution(x_n)
    step_sizes = (0.1, 0.05, 0.025, 0.0125)
    results = []

    print("Ejercicio 1b: intervalo [1, 2] (compatible con las condiciones de frontera).")
    print(" y(1)=3 y y(2)=15; en [3, 4], la misma solución no satisface esos valores.")
    print("| h | Error RMS | Factor RMS previo/actual |")
    print("|---:|---:|---:|")

    # Guarda errores y porcentajes para cada tamaño de paso.
    for h in step_sizes:
        n = int(round((x_n - x_0)/h))
        h = (x_n - x_0)/n
        x = np.linspace(x_0, x_n, n + 1)
        y = diffincentr(x, y_0, y_n, n, h)
        exact = exact_solution(x)
        absolute_error = np.abs(y - exact)
        rms = float(np.sqrt(np.mean(absolute_error**2)))
        true_percent_error = 100*absolute_error/np.abs(exact)
        results.append((h, rms, x, true_percent_error))

    for index, (h, rms, _, _) in enumerate(results):
        factor = "N/A" if index == 0 else f"{results[index-1][1]/rms:.6f}"
        print(f"| {h:.5f} | {rms:.10e} | {factor} |")

    # Comprueba que al reducir el paso a la mitad el error casi se divida por cuatro.
    for index in range(1, len(results)):
        factor = results[index-1][1]/results[index][1]
        if not np.isclose(factor, 4.0, rtol=0.12):
            raise AssertionError(
                f"Convergencia O(h^2) no verificada: factor observado {factor:.6f}."
            )

    # Muestra cuánto se aparta cada punto de su valor conocido.
    print("\nError verdadero porcentual por nodo:")
    for h, _, x, percent_error in results:
        print(f"\nh = {h:.5f}")
        print("| x | Error verdadero porcentual (%) |")
        print("|---:|---:|")
        for x_i, error_i in zip(x, percent_error):
            print(f"| {x_i:.5f} | {error_i:.10e} |")

    print("\nValidación superada: el RMS disminuye aproximadamente por un factor de 4.")


if __name__ == "__main__":
    main()
