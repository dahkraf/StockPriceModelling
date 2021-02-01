# Stock Price Modelling. Stochastic Differential Equations
import numpy as np

def normal_distribution(mean, var, N_samples):
    return np.random.normal(mean, var, N_samples)

def Wiener_process(N, increments):
    Wiener = np.zeros(N)
    for i, element in enumerate(Wiener):
        if(i == 0):
            continue
        Wiener[i] = Wiener[i-1] + increments[i-1]
    return Wiener


def Euler_scheme(xi_0, sigma_0, S_0, coeff, alfa, p, Wiener, delta_t, N):
    xi_array = np.full(N, xi_0)
    sigma_array = np.full(N, sigma_0)
    S_array = np.full(N, S_0)

    for i in range(1, N):
        xi_array[i] = xi_array[i-1] + (1/alfa) * (sigma_array[i-1] - xi_array[i-1]) * delta_t
        sigma_array[i] = sigma_array[i-1] - (sigma_array[i-1] - xi_array[i-1]) * delta_t + p * sigma_array[i-1] * (Wiener[i] - Wiener[i-1])
        S_array[i] = S_array[i-1] + coeff * S_array[i-1] * delta_t + sigma_array[i-1] * S_array[i-1] * (Wiener[i] - Wiener[i-1])

    return xi_array, sigma_array, S_array

# Space Parameters
T_end = 1.0
N = 5
delta_t = T_end/N

# General Equation Parameters
S_0 = 50.0
sigma_0 = 0.20
xi_0 = 0.20
coeff = 0.1

# Variable Parameters
p = 0
alfa = 1

# Wiener process
timestamps = np.linspace(0.0, T_end, N)
increments = normal_distribution(0, delta_t, N-1)
Wiener = Wiener_process(N, increments)

# Numerical solutions
xi_array, sigma_array, S_array = Euler_scheme(xi_0, sigma_0, S_0, coeff, alfa, p, Wiener, delta_t, N)
