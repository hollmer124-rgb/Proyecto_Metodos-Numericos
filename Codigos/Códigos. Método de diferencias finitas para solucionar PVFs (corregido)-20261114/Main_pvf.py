"""Caso base de deflexión de la viga con paso de 6 pulgadas."""

import numpy as np

import deffuncionespvf as dfp


def main():
    # Crea la lista de posiciones con una separación de seis pulgadas.
    h = 6.0
    n = int(round(dfp.L / h))
    x = np.linspace(0.0, dfp.L, n + 1)
    # Evalúa la fórmula exacta y busca el punto de mayor deflexión.
    w = dfp.funcsolex(x)
    maximum_index = int(np.argmax(np.abs(w)))

    # Muestra los resultados en una tabla fácil de comparar.
    print("Caso base: deflexión analítica con w(0) = w(l) = 0")
    print("| x (pulg) | w(x) (pulg) |")
    print("|---:|---:|")
    for x_i, w_i in zip(x, w):
        print(f"| {x_i:.1f} | {w_i:.10e} |")
    print(
        f"\nDeflexión máxima en magnitud: |w_max| = "
        f"{abs(w[maximum_index]):.10e} pulg en "
        f"x_max = {x[maximum_index]:.1f} pulg "
        f"(w = {w[maximum_index]:.10e} pulg)."
    )


if __name__ == "__main__":
    main()
