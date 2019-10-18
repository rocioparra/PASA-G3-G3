"""Filtrado óptimo Wiener con descenso por gradiente.

22.46 Procesamiento adaptativo de Señales Aleatorias
"""

import numpy as np

def steepest_descent(R, p, w0, mu, N):
    """Implementa el filtrado óptimo Wiener con descenso por gradiente.

    Argumentos:
        R: matriz de autocorrelación
        p: matriz de correlación cruzada
        w0: valor inicial de los coeficientes del filtro
        mu: tamaño de paso
        N: número máximo de iteraciones

    Devuelve:
        Una matriz de tipo np.array en cuyas filas están
        los coeficientes w para cada paso.
    """
    Wt = np.zeros((N, len(w0)))

    return Wt
