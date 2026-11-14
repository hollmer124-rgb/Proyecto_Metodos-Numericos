"""Validación y experimentos numéricos de la vibración de la pasarela."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import deffuncionespvi2 as model
from met_solpvi_sist import eul_meth, heun_meth, pm_meth, rk4_meth


INTEGRATORS = (
    ("Euler", eul_meth),
    ("Heun", heun_meth),
    ("Punto Medio", pm_meth),
    ("RK4", rk4_meth),
)
OUTPUT_DIRECTORY = Path(__file__).parent


def validate_workshop():
    """Compara los cuatro integradores con las soluciones exactas del taller."""
    root_1b = np.sqrt(8.0 - 0.3**2)

    def exercise_1a(t, y):
        return np.array([y[1], 0.5 * t - y[0]])

    def exact_1a(t):
        return 2.0 * np.cos(t) - 0.5 * np.sin(t) + 0.5 * t

    def exercise_1b(_t, y):
        _ = _t
        return np.array([y[1], -0.6 * y[1] - 8.0 * y[0]])

    def exact_1b(t):
        return np.exp(-0.3 * t) * (
            4.0 * np.cos(root_1b * t)
            + (1.2 / root_1b) * np.sin(root_1b * t)
        )

    cases = (
        ("1a, [0, 1]", exercise_1a, [2.0, 0.0], 1.0, exact_1a),
        ("1b, [0, 2]", exercise_1b, [4.0, 0.0], 2.0, exact_1b),
    )
    h = 0.01
    print("Validacion del Taller de PVIs (error RMS de y)")
    print("| Ejercicio | Metodo | Error RMS |")
    print("|---|---|---:|")
    for label, funced, initial_state, end_time, exact in cases:
        nit = int(round(end_time / h))
        for method_name, integrator in INTEGRATORS:
            times, states = integrator(funced, 0.0, initial_state, nit, h)
            rms = float(np.sqrt(np.mean((states[:, 0] - exact(times)) ** 2)))
            if states.shape != (nit + 1, 2) or not np.isfinite(rms):
                raise AssertionError(f"Resultado no valido para {label}, {method_name}.")
            if method_name == "RK4" and rms > 1e-6:
                raise AssertionError(f"RK4 no valido para {label}: RMS={rms:.3e}.")
            print(f"| {label} | {method_name} | {rms:.10e} |")


def run_unforced_case():
    """Compara amortiguamientos lineales con F0=0 en 0 <= t <= 15 s."""
    h = 0.01
    nit = int(round(model.T_END / h))
    initial_state = [model.INITIAL_DISPLACEMENT, model.INITIAL_VELOCITY]
    responses = {}

    plt.figure(figsize=(9, 5.5))
    for damping in model.LINEAR_DAMPING_VALUES:
        funced = lambda t, y, c=damping: model.bridge_linear(t, y, c)
        times, states = rk4_meth(funced, 0.0, initial_state, nit, h)
        responses[damping] = (times, states)
        plt.plot(times, states[:, 0], label=f"c = {damping:g} Ns/m")

    plt.xlabel("Tiempo t (s)")
    plt.ylabel("Desplazamiento x(t) (m)")
    plt.title("Pasarela sin fuerza peatonal (F₀ = 0)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = OUTPUT_DIRECTORY / "fase2_caso_base.png"
    plt.savefig(path, dpi=180)
    print(f"\nCaso base sin fuerza: x(0) = {initial_state[0]:g} m, "
          f"v(0) = {initial_state[1]:g} m/s, T = {model.T_END:g} s")
    for damping, (times, states) in responses.items():
        print(
            f"  c={damping:g}: x(T)={states[-1, 0]:.8e} m, "
            f"max|x|={np.max(np.abs(states[:, 0])):.8e} m"
        )
    print(f"Grafica guardada: {path}")
    plt.close()


def run_resonance_sweep():
    """Calcula las amplitudes y refina el máximo de resonancia para c=5."""
    h = 0.02
    linear_damping = 5.0
    resonance_frequency = np.sqrt(
        model.STIFFNESS / model.MASS
        - 2.0 * (linear_damping / (2.0 * model.MASS)) ** 2
    )
    fine_frequencies = np.linspace(
        resonance_frequency - 0.08, resonance_frequency + 0.08, 161
    )
    fine_frequencies = np.append(fine_frequencies, resonance_frequency)
    frequency_grids = {
        damping: model.FREQUENCIES.copy()
        for damping in model.LINEAR_DAMPING_VALUES
    }
    frequency_grids[linear_damping] = np.unique(
        np.concatenate((model.FREQUENCIES, fine_frequencies))
    )
    minimum_frequency = min(grid[0] for grid in frequency_grids.values())
    period_max = 2.0 * np.pi / minimum_frequency
    simulation_end = 50.0 + 5.0 * period_max
    nit = int(round(simulation_end / h))
    simulation_end = nit * h
    initial_state = [model.INITIAL_DISPLACEMENT, model.INITIAL_VELOCITY]
    amplitudes = {}

    for damping in model.LINEAR_DAMPING_VALUES:
        frequencies = frequency_grids[damping]
        curve = []
        for omega in frequencies:
            funced = lambda t, y, c=damping, w=omega: model.bridge_linear(
                t, y, c, model.FORCE_AMPLITUDE, w
            )
            times, states = rk4_meth(funced, 0.0, initial_state, nit, h)
            window_start = simulation_end - 3.0 * (2.0 * np.pi / omega)
            steady_times = times[times >= window_start]
            steady_state = states[times >= window_start, 0]
            harmonic_basis = np.column_stack(
                (
                    np.cos(omega * steady_times),
                    np.sin(omega * steady_times),
                    np.ones_like(steady_times),
                )
            )
            harmonic_coefficients = np.linalg.lstsq(
                harmonic_basis, steady_state, rcond=None
            )[0]
            curve.append(
                float(np.hypot(harmonic_coefficients[0], harmonic_coefficients[1]))
            )
        amplitudes[damping] = (frequencies, np.asarray(curve))

    plt.figure(figsize=(9, 5.5))
    for damping, (frequencies, curve) in amplitudes.items():
        plt.plot(frequencies, curve, "o-", label=f"c = {damping:g} Ns/m")
        peak_index = int(np.argmax(curve))
        if 0 < peak_index < frequencies.size - 1:
            local_fit = np.polyfit(
                frequencies[peak_index - 1:peak_index + 2],
                curve[peak_index - 1:peak_index + 2],
                2,
            )
            omega_peak = -local_fit[1] / (2.0 * local_fit[0])
            amplitude_peak = float(np.polyval(local_fit, omega_peak))
            peak_kind = "pico interior"
        else:
            omega_peak = frequencies[peak_index]
            amplitude_peak = curve[peak_index]
            peak_kind = "maximo en borde; sin pico interior"
        print(
            f"c={damping:g} Ns/m: {peak_kind}, "
            f"omega={omega_peak:.5f} rad/s, "
            f"X_max={amplitude_peak:.8e} m"
        )
        if damping == linear_damping:
            analytic_amplitude = (
                model.FORCE_AMPLITUDE / model.MASS
                / np.sqrt(
                    (model.STIFFNESS / model.MASS - resonance_frequency**2) ** 2
                    + (damping * resonance_frequency / model.MASS) ** 2
                )
            )
            print(
                "  Referencia analitica: "
                f"omega_r={resonance_frequency:.6f} rad/s, "
                f"X_max={analytic_amplitude:.8e} m"
            )
            if not np.isclose(omega_peak, resonance_frequency, atol=1e-3):
                raise AssertionError("El barrido no localizo la resonancia teorica.")
            if not np.isclose(amplitude_peak, analytic_amplitude, rtol=1e-3):
                raise AssertionError(
                    "La amplitud numerica de resonancia no coincide con la analitica."
                )

    plt.xlabel("Frecuencia de forzamiento ω (rad/s)")
    plt.ylabel("Amplitud estacionaria X(ω) (m)")
    plt.title(f"Curvas de resonancia (F₀ = {model.FORCE_AMPLITUDE:g} N)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = OUTPUT_DIRECTORY / "fase2_resonancia.png"
    plt.savefig(path, dpi=180)
    print(
        f"Barrido base: {model.FREQUENCIES.size} frecuencias entre "
        f"{model.FREQUENCIES[0]:g} y {model.FREQUENCIES[-1]:g} rad/s; "
        f"barrido fino de c=5 alrededor de {resonance_frequency:.5f} rad/s "
        f"con {frequency_grids[linear_damping].size} frecuencias; "
        f"amplitud medida en los últimos 3 ciclos tras {simulation_end:.1f} s."
    )
    print(f"Grafica guardada: {path}")
    plt.close()


def run_quadratic_damping_case():
    """Compara amortiguamiento cuadrático puro y combinado en régimen estable."""
    h = 0.005
    simulation_end = 200.0
    omega = 0.9853
    nit = int(round(simulation_end / h))
    times = h * np.arange(nit + 1, dtype=float)
    initial_state = [model.INITIAL_DISPLACEMENT, model.INITIAL_VELOCITY]
    period = 2.0 * np.pi / omega
    window_start = simulation_end - 10.0 * period
    steady_mask = times >= window_start
    cases = (
        ("Amortiguamiento cuadratico puro", 0.0, 1.455),
        ("Amortiguamiento combinado", 5.0, 0.8895),
    )
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    amplitudes = {}

    for axis, (label, linear_damping, expected_amplitude) in zip(axes, cases):
        funced = lambda t, y, c=linear_damping: model.bridge_quadratic(
            t, y, model.FORCE_AMPLITUDE, omega, c
        )
        rk_times, rk_states = rk4_meth(funced, 0.0, initial_state, nit, h)
        reference = model.bridge_nonlinear_reference(
            0.0,
            simulation_end,
            initial_state,
            model.FORCE_AMPLITUDE,
            omega,
            t_eval=times,
            linear_damping=linear_damping,
        )
        rms = float(np.sqrt(np.mean((rk_states[:, 0] - reference.y[0]) ** 2)))
        steady_state = rk_states[steady_mask, 0]
        amplitude = float(0.5 * (np.max(steady_state) - np.min(steady_state)))
        amplitudes[linear_damping] = amplitude
        if not np.isclose(amplitude, expected_amplitude, rtol=3e-3):
            raise AssertionError(
                f"Amplitud estacionaria inesperada para c={linear_damping:g}: "
                f"{amplitude:.6f} m."
            )

        print(
            f"\nVariante C ({label}): c={linear_damping:g} N*s/m, "
            f"c2={model.QUADRATIC_DAMPING:g} N*s^2/m^2, "
            f"F0={model.FORCE_AMPLITUDE:g} N, omega={omega:.4f} rad/s"
        )
        print(f"  Amplitud estacionaria X_max: {amplitude:.8f} m")
        print(f"  Error RMS RK4 frente a solve_ivp(RK45): {rms:.10e} m")
        axis.plot(
            rk_times[steady_mask], rk_states[steady_mask, 0],
            label="RK4 vectorial"
        )
        axis.plot(
            reference.t[steady_mask], reference.y[0, steady_mask], "--",
            label="solve_ivp (RK45)"
        )
        axis.set_ylabel("Desplazamiento x(t) (m)")
        axis.set_title(f"{label} (c = {linear_damping:g} N·s/m)")
        axis.grid(True, alpha=0.3)
        axis.legend()

    linear_only_amplitude = (
        model.FORCE_AMPLITUDE / model.MASS
        / np.sqrt(
            (model.STIFFNESS / model.MASS - omega**2) ** 2
            + (5.0 * omega / model.MASS) ** 2
        )
    )
    reduction_vs_linear = 100.0 * (
        1.0 - amplitudes[5.0] / linear_only_amplitude
    )
    reduction_vs_quadratic = 100.0 * (
        1.0 - amplitudes[5.0] / amplitudes[0.0]
    )
    print(
        "  Reduccion frente al caso lineal c=5 en esta frecuencia: "
        f"{reduction_vs_linear:.2f}%"
    )
    print(
        "  Reduccion frente al caso cuadratico puro: "
        f"{reduction_vs_quadratic:.2f}%"
    )
    axes[-1].set_xlabel("Tiempo t (s)")
    figure.suptitle(
        "Variante C: respuesta estacionaria con amortiguamiento cuadrático"
    )
    figure.tight_layout()
    path = OUTPUT_DIRECTORY / "fase2_variante_c.png"
    figure.savefig(path, dpi=180)
    print(f"Grafica guardada: {path}")
    plt.close()


def main():
    print(
        "Modelo de pasarela: m={:g} kg, k={:g} N/m, "
        "omega_n={:.4f} rad/s".format(
            model.MASS,
            model.STIFFNESS,
            np.sqrt(model.STIFFNESS / model.MASS),
        )
    )
    validate_workshop()
    run_unforced_case()
    run_resonance_sweep()
    run_quadratic_damping_case()


if __name__ == "__main__":
    main()
