"""Prueba de filtrado óptimo Wiener con descenso por gradiente.

22.46 Procesamiento adaptativo de Señales Aleatorias
"""

from steepest_descent import steepest_descent

import math
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.linalg import toeplitz
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Simula un canal
def simulate_channel(y, snr_db):
    w = [1.0000, 2.4156, 2.2226, 0.9578, 0.1884, 0.0130];
    u = signal.filtfilt(1, w, y);

    # Añade ruido blanco
    Pmeas = np.var(y);
    snr = 10 ** (snr_db / 10);
    P_noise = Pmeas / snr;
    additive_white_noise = math.sqrt(P_noise) * np.random.randn(*u.shape);

    # Simula cuantización
    u = np.round(u + additive_white_noise);

    return (u, w)

# Lee archivo de entrada
fs, d = wavfile.read('Tamara_Laurel_-_Sweet_extract.wav')
d = np.float32(d)

# Simula un canal
u, w_true = simulate_channel(d, 80);

# Extrae segmentos de un segundo de ambas señales
s_start = 8;
d = d[s_start * fs:(s_start + 1) * fs];
u = u[s_start * fs:(s_start + 1) * fs];

# Estima la autocorrelación y correlación cruzada
N_THETA = 6;
r = np.correlate(u, u, 'full') / len(u);
r = r[(len(u) - 1):len(u) - 1 + N_THETA];
R = toeplitz(r);

p = np.correlate(d, u, 'full') / len(u);
p = p[(len(u) - 1):len(u) - 1 + N_THETA];

# Determina el filtro óptimo Wiener
w_wiener = inv(R).dot(p);

# Encuentra el filtro óptimo Wiener con descenso por gradiente
mus = [1e-10, 1e-9, 1e-8, 1e-7]; # Diferentes tamaños de paso
w0 = np.zeros(N_THETA);
for mu in mus:
    N = 5000; # Número de iteraciones

    # Llama a la función de filtrado óptimo Wiener con descenso por gradiente.
    # Las filas de Wt representan los filtros en diferentes instantes.
    Wt = steepest_descent(R, p, w0, mu, N);

    # Calcula instante a instante el error cuadrático medio de los
    # coeficientes del filtro respecto de los coeficientes del filtro óptimo.
    mse_coeffs = np.mean((Wt - w_wiener) ** 2, 1)

    plt.plot(mse_coeffs, label='mu=10^%i' % math.log10(mu))

# Representación
plt.xlabel('Número de iteración')
plt.ylabel('MSE de los coeficientes')
plt.title('Filtrado óptimo Wiener con descenso por gradiente')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()
