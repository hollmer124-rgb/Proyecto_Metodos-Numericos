'''
Código construido para solucionar un PVI conformado por una EDO de segundo orden de la forma y'' = fld(t,y) y condiciones iniciales y(t_0)=y_0; y'(t_0)=y'_0.
'''

import numpy as np
import met_solpvi_2ord as pvim2
import deffuncionespvi2 as dfp2
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Información de entrada
# Condiciones iniciales para la vriable independiente
t_0 = 0.
# Condiciones iniciales para la vriable dependiente
y_0 = [4., 1.]

# Punto final del intervalo de solución
T = 3.
# Tamaño de paso
hp = 0.5

# Calculo del número de pasos (ó número de iteraciones)
nit = int(round((T - t_0)/hp))
if nit < 1:
    raise ValueError("El intervalo y el tamaño de paso deben producir al menos un paso.")
hp = (T - t_0)/nit

# Solución PVIs de segundo orden
# =================================
# Método de Euler
teu, yeu = pvim2.euler_methord2(t_0,y_0,nit,hp)
# Método de Heun
the, yhe = pvim2.heun_methord2(t_0,y_0,nit,hp)
# Método de Punto Medio
tptom, yptom = pvim2.ptomed_methord2(t_0,y_0,nit,hp)
# Método de Runge-Kutta de 4to orden
trk4, yrk4 = pvim2.rk4_methord2(t_0,y_0,nit,hp)


# Gráficas
# Cantidad de puntos en la gráfica
cpg = 200
# Generación de puntos en el intervalo [t_0,T]
t_v = np.linspace(t_0,T,cpg)

plt.figure()
plt.plot(dfp2.funcsolref(t_0, T, y_0).t, dfp2.funcsolref(t_0, T, y_0).y[0], 'r', label='Solución de referencia (solve_ivp)')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.show()