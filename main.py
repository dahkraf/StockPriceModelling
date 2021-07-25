# Stock Price Modelling. Stochastic Differential Equations
import numpy as np
import matplotlib.pyplot as plt
import itertools
from matplotlib.font_manager import FontProperties
from matplotlib.colors import hsv_to_rgb
import sys

# Plot settings
plt.rcParams['figure.figsize'] = (9,6)
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False

# Plot styling
# pal = ["#B11412","#B3CDE3", "#CCEBC5","#CFCCC4", "#C6A205", "#0FB379", "#E048A6", "#1A62E2", "#00C9B2", "#E18D13", "#8ADD11"]
pal = ["#294AD9","#0D98DF", "#11E4C6","#0CE359", "#36BC14", "#ABD800", "#F3E80B", "#E89000", "#E4602B", "#D73333", "#DE4CE2"]
def decimal_rgb_to_hex_colour(r, g, b):
    r = "{0:0{1}x}".format(r, 2)
    g = "{0:0{1}x}".format(g, 2)
    b = "{0:0{1}x}".format(b, 2)
    final_color = str.upper("#" + r + g + b)
    return final_color
def generate_continuous_palette(num_of_values, sat, val):
    colour_step = int(256 / num_of_values)

    hues = list(range(0, 256, colour_step))
    hsv_list = list(map((lambda hue: (hue / 256.0, sat / 256.0, val / 256.0)), hues))
    rgb_list = list(map((lambda xyz: tuple(map(lambda a: int(256.0 * a), hsv_to_rgb(xyz)))), hsv_list))

    continuous_palette = list(map(lambda xyz: decimal_rgb_to_hex_colour(*xyz), rgb_list))
    return continuous_palette

# White Noise
def normal_distribution(mean, var, N_samples, seed):
    np.random.seed(seed)
    return np.random.normal(mean, var, N_samples)
def Wiener_process(N, increments):
    Wiener = np.zeros(N)
    for i, element in enumerate(Wiener):
        if(i == 0):
            continue
        Wiener[i] = Wiener[i-1] + increments[i-1]
    return Wiener
def quick_Wiener(N, delta_t, seed):
    np.random.seed(seed)
    dB = np.sqrt(delta_t) * np.random.randn(N)
    return np.cumsum(dB)
def pair_of_Wieners(N, delta_t, seeds):
    increments1 = normal_distribution(0, delta_t, N - 1, seeds[0])
    increments2 = normal_distribution(0, delta_t, N - 1, seeds[1])
    Wiener1 = Wiener_process(N, increments1)
    Wiener2 = Wiener_process(N, increments2)
    return Wiener1, Wiener2
def pair_of_quick_Wieners(N, delta_t, seeds):
    Wiener1 = quick_Wiener(N, delta_t, seeds[0])
    Wiener2 = quick_Wiener(N, delta_t, seeds[1])
    return Wiener1, Wiener2

# Numerical Schemes
def Euler_scheme(xi_0, sigma_0, S_0, coeff, alfa, p, Wiener1, Wiener2, delta_t, N):
    xi_array = np.full(N, xi_0)
    sigma_array = np.full(N, sigma_0)
    S_array = np.full(N, S_0)

    for i in range(1, N):
        xi_array[i] = xi_array[i-1] + (1.0/alfa) * (sigma_array[i-1] - xi_array[i-1]) * delta_t
        sigma_array[i] = sigma_array[i-1] - ((sigma_array[i-1] - xi_array[i-1]) * delta_t) + p * sigma_array[i-1] * (Wiener2[i] - Wiener2[i-1])
        S_array[i] = S_array[i-1] + coeff * S_array[i-1] * delta_t + sigma_array[i-1] * S_array[i-1] * (Wiener1[i] - Wiener1[i-1])

    return xi_array, sigma_array, S_array
def Milstein_scheme(xi_0, sigma_0, S_0, coeff, alfa, p, Wiener1, Wiener2, delta_t, N):
    xi_array = np.full(N, xi_0)
    sigma_array = np.full(N, sigma_0)
    S_array = np.full(N, S_0)

    for i in range(1, N):
        xi_array[i] = xi_array[i-1] + (1.0/alfa) * (sigma_array[i-1] - xi_array[i-1]) * delta_t
        sigma_array[i] = sigma_array[i-1] - (sigma_array[i-1] - xi_array[i-1]) * delta_t + p * sigma_array[i-1] * (Wiener2[i] - Wiener2[i-1]) + 0.5 * p**2 * sigma_array[i-1] * ((Wiener2[i] - Wiener2[i-1])**2 - delta_t)
        S_array[i] = S_array[i-1] + coeff * S_array[i-1] * delta_t + sigma_array[i-1] * S_array[i-1] * (Wiener1[i] - Wiener1[i-1]) + 0.5 * sigma_array[i-1]**2 * S_array[i-1] * ((Wiener1[i] - Wiener1[i-1])**2 - delta_t)

    return xi_array, sigma_array, S_array
def Black_Scholes_exact(sigma_0, coeff, S_0, Wiener1, dt, N, T_end):
    solution = S_0 * np.exp((coeff - 0.5 * sigma_0**2) * T_end + Wiener1[N - 1] * sigma_0)
    return solution

def compare_schemes(N, delta_t, seeds, xi_0, sigma_0, S_0, coeff, alfa, p):
    # Generate Wiener processes
    # Wiener1, Wiener2 = pair_of_Wieners(N, delta_t, seeds)
    Wiener1, Wiener2 = pair_of_quick_Wieners(N, delta_t, seeds)

    # Solve numerically
    _, _, S_array_E = Euler_scheme(xi_0, sigma_0, S_0, coeff, alfa, p, Wiener1, Wiener2, delta_t, N)
    _, _, S_array_M = Milstein_scheme(xi_0, sigma_0, S_0, coeff, alfa, p, Wiener1, Wiener2, delta_t, N)

    # Plot the graphs
    timestamps = np.linspace(0.0, T_end, N)
    plot_schemes(timestamps, S_array_E, S_array_M, alfa, p)

# Plotting
def plot_schemes(timestamps, Euler_array, Milstein_array, alfa, p):
    plt.plot(timestamps, Euler_array, label="Euler ($S_t$)", color=pal[0])
    plt.plot(timestamps, Milstein_array, label="Milstein ($S_t$)", color=pal[1])
    # plt.plot(timestamps, B_S, label="Exact ($S_t$)", color=pal[2])
    plt.title(r"Euler vs. Milstein schemes for $p =$ {p} and $\alpha =$ {alfa}".format(p=p, alfa=alfa))
    plt.xlabel("$t$")
    plt.ylabel("$S_t$")
    plt.legend(loc=2)
    plt.savefig("Plots/Schemes_p_{p}_alpha_{alpha}.jpg".format(p=p, alpha=alfa))
def plot_errors(variable, SE, WE, SM, WM):
    fig, ax = plt.subplots()
    ax.plot(variable, SE, label="Euler Strong Error", color="#AC3015")
    #ax.plot(variable, WE, label="Euler Weak Error", color="#AC3015", ls='--')
    ax.plot(variable, SM, label="Milstein Strong Error", color="#1E97DE")
    #ax.plot(variable, WM, label="Milstein Weak Error", color="#1E97DE", ls='--')
    #ax.set_ylim([(10**-3), (10**1)])
    #ax.set_xlim([min(variable), max(variable)])
    ax.set_xlabel('$\Delta t$')
    ax.set_ylabel('Error$_N$')
    ax.legend()
    plt.show()
def plot_volatility(timestamps, Stock, Volatility, alfa, p):
    plt.plot(timestamps, Stock, label="Stock ($S_t$)", color=pal[0])
    # plt.plot(timestamps, B_S, label="Exact ($S_t$)", color=pal[2])
    plt.title(r"Stock for $p =$ {p} and $\alpha =$ {alfa}".format(p=p, alfa=alfa))
    plt.xlabel('t')
    plt.legend(loc = 2)
    plt.savefig("Stock Price/Stock_p_{p}_alpha_{alpha}.jpg".format(p=p, alpha=alfa))
    plt.close()

    plt.plot(timestamps, Volatility, label="Volatility ($S_t$)", color=pal[1])
    plt.title(r"Volatility for $p =$ {p} and $\alpha =$ {alfa}".format(p=p, alfa=alfa))
    plt.xlabel('t')
    plt.legend(loc=2)
    plt.savefig("Volatility/Vol_p_{p}_alpha_{alpha}.jpg".format(p=p, alpha=alfa))
    plt.close()
def plot_parameter_analysis(timestamps, outputs, parameters, output_variable="S_t", var_of_interest="p"):
    fontP = FontProperties()
    fontP.set_size('x-small')

    # Generate a new palette
    number_of_values = len(parameters)
    sat, val = 185, 213
    continuous_palette = generate_continuous_palette(number_of_values, sat, val)

    fig, ax = plt.subplots()

    # Graph settings
    fig.set_figheight(7)
    fig.set_figwidth(12)
    time_start, time_end = timestamps[0], timestamps[len(timestamps) - 1]
    x_step = (time_end - time_start)/10.0
    ax.xaxis.set_ticks(np.arange(time_start, time_end, x_step))
    ax.set_xlim((time_start, time_end))

    output_variable_latex_name = output_variable
    if (output_variable == "xi"):
        output_variable_latex_name = "$\\" + output_variable_latex_name + "$"
        for i, param in enumerate(parameters):
            p = param[0]
            alfa = param[1]
            ax.plot(timestamps, outputs[i], label="$p =$ {p} $\\alpha =$ {alfa}".format(p=p, alfa=alfa), color=continuous_palette[i], linewidth=1.5)
    elif (output_variable == "sigma"):
        output_variable_latex_name = "$\\" + output_variable_latex_name + "$"
        for i, param in enumerate(parameters):
            p = param[0]
            alfa = param[1]
            ax.plot(timestamps, outputs[i], label="$p =$ {p} $\\alpha =$ {alfa}".format(p=p, alfa=alfa), color=continuous_palette[i], linewidth=1.5)
    else:
        output_variable = "S_t"
        output_variable_latex_name = output_variable
        output_variable_latex_name = "$" + output_variable_latex_name + "$"
        for i, param in enumerate(parameters):
            p = param[0]
            alfa = param[1]
            ax.plot(timestamps, outputs[i], label="$p =$ {p} $\\alpha =$ {alfa}".format(p=p, alfa=alfa), color=continuous_palette[i], linewidth=1.5)

    ax.set_xlabel("$t$")
    ax.set_ylabel(output_variable_latex_name)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', prop=fontP)
    plt.savefig("Parameter Analysis/Param_Analysis_{output_variable}.svg".format(output_variable=output_variable), bbox_inches='tight')
    fig.clf()
    plt.close()

def plot_stock_expectation(p_array, Avg_stocks):
    sat, val = 185, 213
    continuous_palette = generate_continuous_palette(len(p_array), sat, val)

    fig, ax = plt.subplots()
    ax.plot(p_array, Avg_stocks, linestyle="--", marker='o', color=continuous_palette[2], mfc=continuous_palette[9], mec=continuous_palette[9])
    ax.set_xlabel('$p$')
    ax.set_ylabel('$E[S_t]$')
    #plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.savefig("Stock Price/ExpectationOfStock.svg")
    fig.clf()
    plt.close()

# Parameter Analysis
def parameter_analysis(N, delta_t, seeds, xi_0, sigma_0, S_0, coeff, parameters, output_type="S_t", var_of_interest="p"):
    Xis = []
    Sigmas = []
    Stocks = []

    # Generate Wiener processes
    # Wiener1, Wiener2 = pair_of_Wieners(N, delta_t, seeds)
    Wiener1, Wiener2 = pair_of_quick_Wieners(N, delta_t, seeds)

    # Solve for different pairs of parameters
    for param in parameters:
        p = param[0]
        alfa = param[1]

        xi_array_M, sigma_array_M, S_array_M = Milstein_scheme(xi_0, sigma_0, S_0, coeff, alfa, p, Wiener1, Wiener2, delta_t, N)

        Xis.append(xi_array_M)
        Sigmas.append(sigma_array_M)
        Stocks.append(S_array_M)

    # Plot the graph
    timestamps = np.linspace(0.0, T_end, N)
    if(output_type == "xi"):
        plot_parameter_analysis(timestamps, Xis, parameters, output_type)
    elif (output_type == "sigma"):
        plot_parameter_analysis(timestamps, Sigmas, parameters, output_type)
    else:
        output_type = "S_t"
        plot_parameter_analysis(timestamps, Stocks, parameters, output_type)

# Convergence Analysis
def sample_wiener(Wiener, N_benchmark, N):
    stepsize = int((N_benchmark - 1) / (N - 1))
    # 10/5 = 2
    # 0 1 2 3 4 5 6 7 8 9 10
    sampled_Wiener = []
    for i in range(0, N_benchmark, stepsize):
        sampled_Wiener.append(Wiener[i])
    return sampled_Wiener

def strong_convergence(sample_sizes, sim_size):
    # Finest grid
    N_benchmark = (200 * (max(sample_sizes) - 1)) + 1
    # Space Parameters
    T_end = 1.0
    delta_t_benchmark = T_end / N_benchmark

    errors = []
    Benchmark_solutions = []
    Benchmark_Wieners = []
    for i in range(sim_size):
        seeds = [i, i+1]
        # Generate a detailed Wiener process
        Wiener1, Wiener2 = pair_of_quick_Wieners(N_benchmark, delta_t_benchmark, seeds)

        benchmark_solution_EoI = numerical_solution(Wiener1, Wiener2, N_benchmark, delta_t_benchmark, scheme_name="Milstein")
        Benchmark_solutions.append(benchmark_solution_EoI)
        Benchmark_Wieners.append((Wiener1, Wiener2))

    for N in sample_sizes:
        error_dt = strong_error(Benchmark_solutions, Benchmark_Wieners, sim_size, N, N_benchmark, T_end)
        errors.append((N, error_dt))
    return errors
def strong_error(Benchmark_solutions, Benchmark_Wieners, sim_size, N, N_benchmark, T_end):
    Euler_error = 0.0
    Milstein_error = 0.0
    for i in range(sim_size):
        benchmark_solution_EoI = Benchmark_solutions[i]
        Wiener1, Wiener2 = Benchmark_Wieners[i]

        # Sample a smaller Wiener process
        sampled_delta_t = T_end / N
        sample_Wiener1 = sample_wiener(Wiener1, N_benchmark, N)
        sample_Wiener2 = sample_wiener(Wiener2, N_benchmark, N)

        # Calculate the solutions at the EoI
        Num_solution_Euler_EoI = numerical_solution(sample_Wiener1, sample_Wiener2, N, sampled_delta_t, "Euler")
        Num_solution_Milstein_EoI = numerical_solution(sample_Wiener1, sample_Wiener2, N, sampled_delta_t, "Milstein")

        # Individual errors
        single_error_Euler = abs(benchmark_solution_EoI - Num_solution_Euler_EoI)
        single_error_Milstein = abs(benchmark_solution_EoI - Num_solution_Milstein_EoI)

        # Accumulate errors
        Euler_error += single_error_Euler
        Milstein_error += single_error_Milstein
    Euler_error = Euler_error / (sim_size * 1.0)
    Milstein_error = Milstein_error / (sim_size * 1.0)
    return Euler_error, Milstein_error

def weak_convergence(sample_sizes, sim_size):
    # Finest grid
    N = 10**3 * max(sample_sizes)

    errors = []
    for n in sample_sizes:
        error_dt = weak_error(sim_size, n, N)
        errors.append((n, error_dt))
    return errors
def weak_error(sim_size, N, N_benchmark):
    Euler_error = 0.0
    Milstein_error = 0.0
    avg_Euler = 0.0
    avg_Milstein = 0.0
    avg_Benchmark = 0.0
    for i in range(sim_size):
        seeds = [i, i + 1]
        benchmark_solution_EoI = numerical_solution(N_benchmark, seeds, scheme_name="Milstein")[N_benchmark - 1]
        Num_solution_Euler = numerical_solution(N, seeds, "Euler")
        Num_solution_Milstein = numerical_solution(N, seeds, "Milstein")
        Euler_EoI = Num_solution_Euler[N - 1]
        Milstein_EoI = Num_solution_Milstein[N - 1]
        avg_Euler += Euler_EoI
        avg_Milstein += Milstein_EoI
        avg_Benchmark += benchmark_solution_EoI
    avg_Euler = avg_Euler / sim_size
    avg_Milstein = avg_Milstein / sim_size
    avg_Benchmark = avg_Benchmark / sim_size
    Euler_error = abs(avg_Benchmark - avg_Euler)
    Milstein_error = abs(avg_Benchmark - avg_Milstein)
    return Euler_error, Milstein_error

def numerical_solution(Wiener1, Wiener2, N, delta_t, scheme_name):
    # General Equation Parameters
    S_0 = 50.0
    sigma_0 = 0.20
    xi_0 = 0.20
    coeff = 0.1
    p = 1.0
    alfa = 1.0

    if (scheme_name == "Milstein"):
        return Milstein_scheme(xi_0, sigma_0, S_0, coeff, alfa, p, Wiener1, Wiener2, delta_t, N)[2][N - 1]
    elif (scheme_name == "BS"):
        return Black_Scholes_exact(sigma_0, coeff, S_0, Wiener1, delta_t, N, T_end)
    return Euler_scheme(xi_0, sigma_0, S_0, coeff, alfa, p, Wiener1, Wiener2, delta_t, N)[2][N - 1]


def convergence_analysis():
    # Grid and simulation size settings
    dt_grid = [10 ** (R-3) for R in range(4)]
    #dt_grid = np.arange(min(dt_grid), max(dt_grid), 10**(-2))
    # minval = 2 ** (-20)
    # dt_grid.insert(0, minval)
    sample_sizes = [(int(1.0 / dt) + 1) for dt in dt_grid]
    dt_new = [(1.0 / ss) for ss in sample_sizes]
    sim_size = 30

    # Calculate errors
    Strong_errors = list(zip(*list(zip(*strong_convergence(sample_sizes, sim_size)))[1]))
    # Weak_errors = list(zip(*list(zip(*weak_convergence(sample_sizes, sim_size)))[1]))
    Strong_Euler_errors = Strong_errors[0]
    Strong_Milstein_errors = Strong_errors[1]
    # Weak_Euler_errors = Weak_errors[0]
    # Weak_Milstein_errors = Weak_errors[1]

    # Plot the error graph
    plot_errors(dt_new, Strong_Euler_errors, [], Strong_Milstein_errors, [])

# N = 13
# n = 7
# delta_t = 1.0/N
# seed = 6
# W = quick_Wiener(N, delta_t, n)
# print(W)
# W_s = sample_wiener(W, N, seed)
# print(W_s)
# sys.exit()
###############################################################################

# Space Parameters
T_end = 1.0
N = 365
delta_t = T_end/N

# General Equation Parameters
S_0 = 50.0
sigma_0 = 0.20
xi_0 = 0.20
coeff = 0.1

# Variable Parameters
# p_array = [-1.0, -8.0, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 0.8, 1.0]
p_array = [-1.2, -1.1, -1.0, -0.95, -0.85, -0.8, -0.75, -0.65, -0.6, -0.5, -0.4]
alfa_array = [1.0]

# for i in range(0, 5):
#     p_array.append(2**i)
#     alfa_array.append(2**i)

# Combine parameters p and alpha (Cartesian product of the two lists)
parameter_combinations = list(itertools.product(p_array, alfa_array))

# Analyse p value
seeds = [0, 1]
parameter_analysis(N, delta_t, seeds, xi_0, sigma_0, S_0, coeff, parameter_combinations, output_type="S_t")
sys.exit()

# Compare schemes
# for pair in parameter_combinations:
#     # Numerical solutions
#     alfa = pair[0]
#     p = pair[1]

sys.exit()
######################################################
# Effect of p on the E[S_t]
# Avg_stocks = []
# array = []
# print(list(parameter_combinations))
# for pair in parameter_combinations:
#     Stocks = 0.0
#     print("here")
#     # Numerical solutions
#     alfa = pair[0]
#     p = pair[1]
#     array.append(p)
#     size = 20
#
#     for i in range(size):
#         print(i)
#         seeds = [i, i + 1]
#         # Wiener processes
#         timestamps = np.linspace(0.0, T_end, N)
#         Wiener1 = Quick_Wiener(N, delta_t, seeds[0])
#         Wiener2 = Quick_Wiener(N, delta_t, seeds[1])
#
#         _, _, S_array_M = Milstein_scheme(xi_0, sigma_0, S_0, coeff, alfa, p, Wiener1, Wiener2, delta_t, N)
#         Stocks += S_array_M[N - 1]
#
#     Stocks /= size
#     Avg_stocks.append(Stocks)
#
# plot_stock_expectation(array, Avg_stocks)

