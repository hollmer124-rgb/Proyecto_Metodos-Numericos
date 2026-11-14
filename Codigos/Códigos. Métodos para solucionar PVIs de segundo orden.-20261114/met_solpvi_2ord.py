"""Reúne cuatro formas de calcular paso a paso una curva y su velocidad."""

import numpy as np
import deffuncionespvi2 as dfp2

def euler_methord2(t0, yo2, n, h):
    """Calcula la posición y la velocidad avanzando de forma sencilla."""
    
    # Prepara las listas de tiempo, posición y velocidad.
    t = np.zeros(n+1)
    y = np.zeros((2,n+1))
    t[0] = t0
    y[0,0] = yo2[0]
    y[1,0] = yo2[1]
    # Usa el valor actual para estimar el siguiente punto.
    for i in range(n):
        y[0,i+1] = y[0,i] + h*y[1,i]
        y[1,i+1] = y[1,i] + h*dfp2.funced(t[i], y[:,i])
        t[i+1] = t[i] + h
    return t,y

def heun_methord2(t0, yo2, n, h):
    """Mejora cada avance usando el promedio entre el inicio y el final."""
    # Guarda la condición inicial antes de empezar los avances.
    t = np.zeros(n+1)
    y = np.zeros((2,n+1))
    t[0] = t0
    y[0,0] = yo2[0]
    y[1,0] = yo2[1]

    for i in range(n):
        # Hace una primera estimación del siguiente punto.
        y_p = np.zeros(2)
        y_p[0] = y[0,i] + h*y[1,i]
        y_p[1] = y[1,i] + h*dfp2.funced(t[i], y[:,i])

        # Ajusta la estimación usando la pendiente promedio.
        f_i = dfp2.funced(t[i], y[:,i])
        f_p = dfp2.funced(t[i] + h, y_p)

        y[0,i+1] = y[0,i] + h*(y[1,i] + y_p[1]) / 2.0
        y[1,i+1] = y[1,i] + h*(f_i + f_p) / 2.0
        t[i+1] = t[i] + h

    return t,y

def ptomed_methord2(t0, yo2, n, h):
    """Calcula cada avance usando una estimación a mitad del paso."""
    # Reserva espacio para guardar la curva calculada.
    t = np.zeros(n+1)
    y = np.zeros((2,n+1))
    t[0] = t0
    y[0,0] = yo2[0]
    y[1,0] = yo2[1]

    for i in range(n):
        f_i = dfp2.funced(t[i], y[:,i])

        # Estima la posición y la velocidad a mitad del paso.
        y_mid = np.zeros(2)
        y_mid[0] = y[0,i] + h*y[1,i]/2.0
        y_mid[1] = y[1,i] + h*f_i/2.0
        t_mid = t[i] + h/2.0

        # Usa esa estimación intermedia para completar el avance.
        f_mid = dfp2.funced(t_mid, y_mid)
        y[0,i+1] = y[0,i] + h*y_mid[1]
        y[1,i+1] = y[1,i] + h*f_mid
        t[i+1] = t[i] + h

    return t,y

def rk4_methord2(t0, yo2, n, h):
    """Calcula cada avance con cuatro estimaciones para ganar precisión."""
    # Reserva espacio para los tiempos y los resultados.
    t = np.zeros(n+1)
    y = np.zeros((2,n+1))
    t[0] = t0
    y[0,0] = yo2[0]
    y[1,0] = yo2[1]

    for i in range(n):
        # Calcula cuatro posibles avances y los combina al final.
        k1 = np.zeros(2)
        k1[0] = y[1,i]
        k1[1] = dfp2.funced(t[i], y[:,i])

        y_k2 = y[:,i] + h*k1/2.0
        k2 = np.zeros(2)
        k2[0] = y_k2[1]
        k2[1] = dfp2.funced(t[i] + h/2.0, y_k2)

        y_k3 = y[:,i] + h*k2/2.0
        k3 = np.zeros(2)
        k3[0] = y_k3[1]
        k3[1] = dfp2.funced(t[i] + h/2.0, y_k3)

        y_k4 = y[:,i] + h*k3
        k4 = np.zeros(2)
        k4[0] = y_k4[1]
        k4[1] = dfp2.funced(t[i] + h, y_k4)

        # Guarda el nuevo punto de la curva y su tiempo.
        y[:,i+1] = y[:,i] + h*(k1 + 2*k2 + 2*k3 + k4)/6.0
        t[i+1] = t[i] + h

    return t,y