"""Resuelve un ejemplo de curva con posición y velocidad iniciales conocidas."""

import numpy as np
import met_solpvi_2ord as pvim2
import deffuncionespvi2 as dfp2
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Indica cuándo empieza el cálculo.
t_0 = 0.
# Fija la posición y la velocidad iniciales.
y_0 = [4., 1.]

# Indica cuándo termina el cálculo y cuánto avanza cada paso.
T = 3.
hp = 0.5

# Ajusta el paso para que el último avance llegue justo al final.
nit = int(round((T - t_0)/hp))
if nit < 1:
    raise ValueError("El intervalo y el tamaño de paso deben producir al menos un paso.")
hp = (T - t_0)/nit

# Calcula la curva con cuatro maneras distintas de avanzar.
teu, yeu = pvim2.euler_methord2(t_0,y_0,nit,hp)
the, yhe = pvim2.heun_methord2(t_0,y_0,nit,hp)
tptom, yptom = pvim2.ptomed_methord2(t_0,y_0,nit,hp)
trk4, yrk4 = pvim2.rk4_methord2(t_0,y_0,nit,hp)


# Prepara puntos para mostrar la curva de referencia del ejemplo.
cpg = 200
t_v = np.linspace(t_0,T,cpg)

# Dibuja la solución de referencia calculada por SciPy.
plt.figure()
plt.plot(dfp2.funcsolref(t_0, T, y_0).t, dfp2.funcsolref(t_0, T, y_0).y[0], 'r', label='Solución de referencia (solve_ivp)')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.show()