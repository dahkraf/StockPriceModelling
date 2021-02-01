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

# Parameters
T_end = 2.0
N = 5
delta_t = T_end/N

timestamps = np.linspace(0.0, T_end, N)
increments = normal_distribution(0, delta_t, N-1)
Wiener = Wiener_process(N, increments)
