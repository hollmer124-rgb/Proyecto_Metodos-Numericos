"""Convergencia de los métodos y límite de estabilidad de Euler."""

from pathlib import Path
import importlib.util
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np

from met_disparo import disparo_lineal
from met_solpvi_sist import eul_meth, heun_meth, pm_meth, rk4_meth
import deffuncionespvi2 as oscillator


# Ubica los datos, los pasos de prueba y las imágenes de salida.
PVF_DIRECTORY = (
    Path(__file__).parent.parent
    / "Códigos. Método de diferencias finitas para solucionar PVFs (corregido)-20261114"
)
OUTPUT_DIRECTORY = Path(__file__).parent
PVF_STEPS = np.array([24.0, 12.0, 6.0, 3.0, 1.5])
PVI_STEPS = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])
# Elige pasos a ambos lados del límite de cada nivel de fricción.
STABILITY_STEP_FACTORS = {
    5.0: (0.80, 0.96, 1.00, 1.04, 1.20),
    40.0: (0.80, 0.95, 1.00, 1.05, 1.20),
    200.0: (0.80, 0.94, 1.00, 1.02, 1.04),
}
# Include the requested comparison value; the computed threshold is about 0.202041 s.
STABILITY_EXTRA_STEPS = {200.0: (0.2021,)}
# Ajusta el tiempo mostrado para que el crecimiento sea visible.
STABILITY_ENDS = {5.0: 60.0, 40.0: 20.0, 200.0: 10.0}
PVI_END = 15.0


def load_pvf_module(module_name) -> ModuleType:
    """Abre un módulo de la viga guardado en la carpeta vecina."""
    module_path = PVF_DIRECTORY / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el módulo {module_path}.")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


beam = load_pvf_module("deffuncionespvf")
finite_difference_module = load_pvf_module("metdf_solpvf")
diffincentr = finite_difference_module.diffincentr

PVI_METHODS = (
    ("Euler", eul_meth, 1),
    ("Heun", heun_meth, 2),
    ("Punto Medio", pm_meth, 2),
    ("RK4", rk4_meth, 4),
)


def rms(first, second):
    """Resume en un valor cuánto se apartan dos listas de resultados."""
    return float(np.sqrt(np.mean((np.asarray(first) - np.asarray(second)) ** 2)))


def observed_order(step_sizes, errors):
    """Estima cómo cambia el error cuando se reduce el tamaño del paso."""
    if np.any(np.asarray(errors) <= 0.0):
        raise ValueError("El ajuste log-log requiere errores RMS positivos.")
    return float(np.polyfit(np.log10(step_sizes), np.log10(errors), 1)[0])


def finite_difference_base(x, intervals, h):
    """Calcula la viga de rigidez fija y luego restaura sus fórmulas originales."""
    # Guarda los datos para poder volver al modelo de rigidez variable.
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
        return diffincentr(x, 0.0, 0.0, intervals, h)
    finally:
        beam.A, beam.B, beam.C, beam.f = original


def beam_rhs(x, state):
    """Describe cómo cambian la altura y la pendiente de la viga."""
    second_derivative = (
        beam.S * state[0] + beam.Q * x * (x - beam.L) / 2.0
    ) / (beam.E * beam.I0)
    return np.array([state[1], second_derivative])


def run_beam_convergence(axis):
    """Compara dos métodos de la viga al usar pasos cada vez más cortos."""
    df_errors = []
    shooting_errors = []
    # Calcula el error en cada paso y conserva una malla común.
    for requested_h in PVF_STEPS:
        intervals = int(round(beam.L / requested_h))
        h = beam.L / intervals
        x = np.linspace(0.0, beam.L, intervals + 1)
        exact = beam.funcsolex(x)

        finite_difference = finite_difference_base(x, intervals, h)
        shot_x, shooting = disparo_lineal(
            beam_rhs, 0.0, beam.L, 0.0, 0.0, h
        )
        if not np.allclose(x, shot_x):
            raise AssertionError("Las mallas de PVF no coinciden.")
        df_errors.append(rms(finite_difference, exact))
        shooting_errors.append(rms(shooting, exact))

    df_errors = np.asarray(df_errors)
    shooting_errors = np.asarray(shooting_errors)
    # Excluye del ajuste de disparo el error más pequeño, dominado por redondeo.
    df_order = observed_order(PVF_STEPS, df_errors)
    shooting_order = observed_order(PVF_STEPS[:-1], shooting_errors[:-1])

    print("\nConvergencia PVF (RMS contra la solución exacta, pulg)")
    print("| Método | " + " | ".join(f"h={h:g}" for h in PVF_STEPS) + " | p observado |")
    print("|---|" + "---:|" * (len(PVF_STEPS) + 1))
    print(
        "| Diferencias finitas | "
        + " | ".join(f"{e:.4e}" for e in df_errors)
        + f" | {df_order:.4f} |"
    )
    print(
        "| Disparo RK4 | "
        + " | ".join(f"{e:.4e}" for e in shooting_errors)
        + f" | {shooting_order:.4f} |"
    )
    print(
        "Nota: el ajuste de Disparo RK4 excluye h=1.5 pulg por estar en el "
        "límite de precisión de la computadora."
    )

    # Dibuja los errores y líneas guía para comparar el ritmo de mejora.
    axis.loglog(PVF_STEPS, df_errors, "o-", label=f"Diferencias finitas (p={df_order:.2f})")
    axis.loglog(
        PVF_STEPS,
        shooting_errors,
        "s-",
        label=f"Disparo RK4 (p={shooting_order:.2f}; ajuste h≥3)",
    )
    for expected_order, label in ((2, "Referencia O(h²)"), (4, "Referencia O(h⁴)")):
        reference = df_errors[0] * (PVF_STEPS / PVF_STEPS[0])**expected_order
        axis.loglog(PVF_STEPS, reference, "--", alpha=0.6, label=label)
    axis.set_xlabel("Paso h (pulg)")
    axis.set_ylabel("Error RMS (pulg)")
    axis.set_title("Convergencia del PVF de la viga")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    axis.invert_xaxis()
    return df_errors, shooting_errors, df_order, shooting_order


def exact_oscillator(t):
    """Devuelve la curva conocida para la pasarela sin fuerza externa."""
    alpha = oscillator.LINEAR_DAMPING_VALUES[0] / (2.0 * oscillator.MASS)
    omega_d = np.sqrt(oscillator.STIFFNESS / oscillator.MASS - alpha**2)
    x0 = oscillator.INITIAL_DISPLACEMENT
    v0 = oscillator.INITIAL_VELOCITY
    return np.exp(-alpha * t) * (
        x0 * np.cos(omega_d * t)
        + ((v0 + alpha * x0) / omega_d) * np.sin(omega_d * t)
    )


def run_oscillator_convergence(axis):
    """Compara cuatro métodos para pasos cada vez más cortos."""
    errors = {name: [] for name, _, _ in PVI_METHODS}
    for h in PVI_STEPS:
        intervals = int(round(PVI_END / h))
        effective_h = PVI_END / intervals
        exact_times = np.linspace(0.0, PVI_END, intervals + 1)
        exact_values = exact_oscillator(exact_times)
        funced = lambda t, state: oscillator.bridge_linear(
            t, state, oscillator.LINEAR_DAMPING_VALUES[0]
        )

        for name, integrator, _ in PVI_METHODS:
            times, states = integrator(
                funced,
                0.0,
                [oscillator.INITIAL_DISPLACEMENT, oscillator.INITIAL_VELOCITY],
                intervals,
                effective_h,
            )
            if not np.allclose(times, exact_times):
                raise AssertionError("La malla de convergencia PVI no coincide.")
            errors[name].append(rms(states[:, 0], exact_values))

    # Resume los errores y el ritmo de mejora de cada método.
    orders = {
        name: observed_order(PVI_STEPS, method_errors)
        for name, method_errors in errors.items()
    }

    print("\nConvergencia PVI (RMS del desplazamiento, m)")
    print("| Método | " + " | ".join(f"h={h:g}" for h in PVI_STEPS) + " | p observado |")
    print("|---|" + "---:|" * (len(PVI_STEPS) + 1))
    # Dibuja los errores y guías que muestran el comportamiento esperado.
    for name, _, _ in PVI_METHODS:
        print(
            f"| {name} | "
            + " | ".join(f"{e:.4e}" for e in errors[name])
            + f" | {orders[name]:.4f} |"
        )

    for name, _, _ in PVI_METHODS:
        axis.loglog(PVI_STEPS, errors[name], "o-", label=f"{name} (p={orders[name]:.2f})")
    reference_sources = {1: "Euler", 2: "Heun", 4: "RK4"}
    for expected_order, source in reference_sources.items():
        reference = (
            errors[source][0]
            * (PVI_STEPS / PVI_STEPS[0])**expected_order
        )
        axis.loglog(
            PVI_STEPS, reference, "--", alpha=0.35,
            label=f"Referencia O(h^{expected_order})"
        )
    axis.set_xlabel("Paso h (s)")
    axis.set_ylabel("Error RMS del desplazamiento (m)")
    axis.set_title("Convergencia del PVI del tablero (c = 5 N·s/m)")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize="small", ncol=2)
    axis.invert_xaxis()
    return errors, orders


def run_euler_stability():
    """Muestra cómo cambia la respuesta de Euler al acercarse al límite."""
    mass = oscillator.MASS
    stiffness = oscillator.STIFFNESS
    damping_values = oscillator.LINEAR_DAMPING_VALUES
    figure, axes = plt.subplots(
        len(damping_values), 1, figsize=(10, 13), constrained_layout=True
    )
    initial_state = [oscillator.INITIAL_DISPLACEMENT, oscillator.INITIAL_VELOCITY]
    print("\nEstabilidad de Euler para los tres regímenes de amortiguamiento")

    # Revisa por separado cada nivel de fricción y sus valores límite.
    for axis, damping in zip(axes, damping_values):
        system_matrix = np.array(
            [
                [0.0, 1.0],
                [-stiffness / mass, -damping / mass],
            ]
        )
        # Usa las raíces del modelo para calcular el paso límite.
        eigenvalues = np.linalg.eigvals(system_matrix)
        critical_candidates = (
            -2.0 * eigenvalue.real / abs(eigenvalue) ** 2
            for eigenvalue in eigenvalues
        )
        critical_h = float(min(critical_candidates))
        critical_damping = 2.0 * np.sqrt(stiffness * mass)
        repeated_root = np.isclose(damping, critical_damping, rtol=1e-12)
        eigenvalue_text = ", ".join(
            f"{eigenvalue.real:.6f}{eigenvalue.imag:+.6f}i"
            if abs(eigenvalue.imag) > 1e-10
            else f"{eigenvalue.real:.6f}"
            for eigenvalue in eigenvalues
        )
        print(
            f"\nc={damping:g} N·s/m; autovalores={eigenvalue_text} 1/s; "
            f"h_critico={critical_h:.5f} s"
        )
        print("| h (s) | rho(I + hA) | Diagnóstico |")
        print("|---:|---:|---|")

        h_values = sorted(
            [critical_h * factor for factor in STABILITY_STEP_FACTORS[damping]]
            + list(STABILITY_EXTRA_STEPS.get(damping, ()))
        )
        # Compara la respuesta con pasos menores, iguales y mayores al límite.
        for h in h_values:
            amplification = float(np.max(np.abs(1.0 + h * eigenvalues)))
            at_threshold = np.isclose(
                amplification, 1.0, rtol=0.0, atol=1e-10
            )
            if at_threshold and repeated_root:
                diagnosis = "umbral; raíz doble, crecimiento polinomial"
                label_state = "umbral, raíz doble"
            elif at_threshold:
                diagnosis = "umbral espectral (neutro)"
                label_state = "umbral"
            elif amplification < 1.0:
                diagnosis = "estable, amplitud decreciente"
                label_state = "estable"
            else:
                diagnosis = "inestable, amplitud creciente"
                label_state = "inestable"
            print(f"| {h:.5f} | {amplification:.8f} | {diagnosis} |")

            end_time = STABILITY_ENDS[damping]
            intervals = int(np.floor(end_time / h))
            # Guarda la trayectoria para ver si se reduce o crece con el tiempo.
            times, states = eul_meth(
                lambda t, state, c=damping: oscillator.bridge_linear(
                    t, state, c
                ),
                0.0,
                initial_state,
                intervals,
                h,
            )
            axis.plot(
                times,
                states[:, 0],
                label=f"h={h:.5g} s ({label_state})",
            )

        axis.set_ylabel("Desplazamiento x(t) (m)")
        axis.set_title(
            f"c = {damping:g} N·s/m; h crítico = {critical_h:.5f} s"
        )
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small", ncol=2)

    axes[-1].set_xlabel("Tiempo t (s)")
    figure.savefig(OUTPUT_DIRECTORY / "fase4_estabilidad.png", dpi=180)


def main():
    # Ejecuta ambos estudios, guarda sus figuras y las muestra al usuario.
    print("Ordenes teoricos: diferencias finitas O(h^2), disparo RK4 O(h^4); "
          "PVI Euler O(h), Heun y punto medio O(h^2), RK4 O(h^4).")
    convergence_figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    run_beam_convergence(axes[0])
    run_oscillator_convergence(axes[1])
    convergence_figure.tight_layout()
    convergence_figure.savefig(
        OUTPUT_DIRECTORY / "fase4_convergencia.png", dpi=180
    )
    run_euler_stability()
    plt.show()


if __name__ == "__main__":
    main()
