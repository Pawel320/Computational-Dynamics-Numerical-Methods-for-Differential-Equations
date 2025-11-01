import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#
# def pendulum(t, y, g=9.81, l=1.0):
#     theta, omega = y
#     dtheta_dt = omega
#     domega_dt = - (g / l) * np.sin(theta)
#     return np.array([dtheta_dt, domega_dt])
#
# # Metoda Eulera
# def euler(f, y0, t):
#     y = np.zeros((len(t), len(y0)))
#     y[0] = y0
#     for i in range(len(t)-1):
#         h = t[i+1] - t[i]
#         y[i+1] = y[i] + h * f(t[i], y[i])
#     return y
#
# # Metoda RK4
# def rk4(f, y0, t):
#     y = np.zeros((len(t), len(y0)))
#     y[0] = y0
#     for i in range(len(t)-1):
#         h = t[i+1] - t[i]
#         k1 = f(t[i], y[i])
#         k2 = f(t[i] + h/2, y[i] + h/2 * k1)
#         k3 = f(t[i] + h/2, y[i] + h/2 * k2)
#         k4 = f(t[i] + h, y[i] + h * k3)
#         y[i+1] = y[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
#     return y
#
# # Obliczanie energii
# def compute_energy(theta, omega, g=9.81, l=1.0, m=1.0):
#     KE = 0.5 * m * (l**2) * omega**2
#     PE = m * g * l * (1 - np.cos(theta))
#     return KE + PE
#
# # === Parametry symulacji ===
# g = 9.81         # [m/s²] przyspieszenie ziemskie
# l = 1.0          # [m] długość wahadła
# theta0 = np.pi / 4  # [rad] 45 stopni
# omega0 = 0          # [rad/s]
# y0 = [theta0, omega0]
#
# t0 = 0
# tf = 10
# dt = 0.01
# t = np.arange(t0, tf + dt, dt)
# steps = len(t)
#
# # === Obliczenia ===
# y_euler = euler(lambda t, y: pendulum(t, y, g, l), y0, t)
# y_rk4 = rk4(lambda t, y: pendulum(t, y, g, l), y0, t)
#
# # === Wykres 1: Rozwiązanie numeryczne ===
# plt.figure(figsize=(10, 6))
# plt.plot(t, y_euler[:, 0], label='Euler', linestyle='--', color='red')
# plt.plot(t, y_rk4[:, 0], label='RK4', linestyle='-', color='blue')
#
# plt.title(f"Wahadło matematyczne\n"
#           f"Równanie: θ'' = -(g/l)·sin(θ), g={g} m/s², l={l} m\n"
#           f"Warunki początkowe: θ(0)={theta0:.3f} rad, ω(0)={omega0:.3f} rad/s\n"
#           f"Krok czasowy: Δt={dt}, liczba kroków: {steps}",
#           fontsize=10)
#
# plt.xlabel("Czas [s]")
# plt.ylabel("Kąt [rad]")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# # === Obliczenie energii ===
# energy_rk4 = compute_energy(y_rk4[:, 0], y_rk4[:, 1], g, l)
# energy_euler = compute_energy(y_euler[:, 0], y_euler[:, 1], g, l)
#
# # === Wykres 2: Energia mechaniczna ===
# plt.figure(figsize=(10, 5))
# plt.plot(t, energy_rk4, label="Energia (RK4)", linestyle="-", color="blue")
# plt.plot(t, energy_euler, label="Energia (Euler)", linestyle="--", color="red")
#
# plt.title("Zachowanie energii mechanicznej w czasie")
# plt.xlabel("Czas [s]")
# plt.ylabel("Energia [J]")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
#
# # Define the pendulum system
# def pendulum(t, y, g=9.81, l=1.0):
#     theta, omega = y
#     dtheta_dt = omega
#     domega_dt = - (g / l) * np.sin(theta)
#     return np.array([dtheta_dt, domega_dt])
#
# # Euler method
# def euler(f, y0, t):
#     y = np.zeros((len(t), len(y0)))
#     y[0] = y0
#     for i in range(len(t) - 1):
#         h = t[i + 1] - t[i]
#         y[i + 1] = y[i] + h * f(t[i], y[i])
#     return y
#
# # RK4 method
# def rk4(f, y0, t):
#     y = np.zeros((len(t), len(y0)))
#     y[0] = y0
#     for i in range(len(t) - 1):
#         h = t[i + 1] - t[i]
#         k1 = f(t[i], y[i])
#         k2 = f(t[i] + h / 2, y[i] + h / 2 * k1)
#         k3 = f(t[i] + h / 2, y[i] + h / 2 * k2)
#         k4 = f(t[i + 1], y[i] + h * k3)
#         y[i + 1] = y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
#     return y
#
# # Parameters
# g = 9.81
# l = 1.0
# theta0 = np.pi / 4  # Initial angle (45 degrees)
# omega0 = 0          # Initial angular velocity
# y0 = [theta0, omega0]
# t0, tf = 0, 10      # Time range
# h_rk4 = 0.01        # Fixed time step for RK4
# t_rk4 = np.arange(t0, tf + h_rk4, h_rk4)
#
# # RK4 solution
# y_rk4 = rk4(lambda t, y: pendulum(t, y, g, l), y0, t_rk4)
#
# # Function to compare Euler with RK4
# def compare_euler_with_rk4(h_euler):
#     t_euler = np.arange(t0, tf + h_euler, h_euler)
#     y_euler = euler(lambda t, y: pendulum(t, y, g, l), y0, t_euler)
#
#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.plot(t_rk4, y_rk4[:, 0], label=f'RK4 (h={h_rk4})', color='blue', linestyle='-')
#     plt.plot(t_euler, y_euler[:, 0], label=f'Euler (h={h_euler})', color='red', linestyle='--')
#     plt.title("Comparison of Euler and RK4 Methods")
#     plt.xlabel("Time [s]")
#     plt.ylabel("Angle [rad]")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
# # Example usage: Change h_euler to test different time steps
# h_euler = 0.01  # Set the time step for Euler
# compare_euler_with_rk4(h_euler)

import numpy as np
import matplotlib.pyplot as plt

# Define the pendulum system
def pendulum(t, y, g=9.81, l=1.0):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = - (g / l) * np.sin(theta)
    return np.array([dtheta_dt, domega_dt])

# Euler method
def euler(f, y0, t):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        y[i + 1] = y[i] + h * f(t[i], y[i])
    return y

# RK4 method
def rk4(f, y0, t):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 2, y[i] + h / 2 * k1)
        k3 = f(t[i] + h / 2, y[i] + h / 2 * k2)
        k4 = f(t[i + 1], y[i] + h * k3)
        y[i + 1] = y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y

# Compute total energy
def compute_energy(theta, omega, g=9.81, l=1.0, m=1.0):
    KE = 0.5 * m * (l**2) * omega**2  # Kinetic energy
    PE = m * g * l * (1 - np.cos(theta))  # Potential energy
    return KE + PE

# Parameters
g = 9.81
l = 1.0
theta0 = np.pi / 4  # Initial angle (45 degrees)
omega0 = 0          # Initial angular velocity
y0 = [theta0, omega0]
t0, tf = 0, 10      # Time range
h_rk4 = 0.01        # Fixed time step for RK4
t_rk4 = np.arange(t0, tf + h_rk4, h_rk4)

# RK4 solution
y_rk4 = rk4(lambda t, y: pendulum(t, y, g, l), y0, t_rk4)

# Function to compare Euler with RK4 and plot energy
def compare_euler_with_rk4_and_energy(h_euler):
    t_euler = np.arange(t0, tf + h_euler, h_euler)
    y_euler = euler(lambda t, y: pendulum(t, y, g, l), y0, t_euler)

    # Interpolate Euler solution to match RK4 time points
    y_euler_interp = np.interp(t_rk4, t_euler, y_euler[:, 0])

    # Calculate maximum error
    max_error = np.max(np.abs(y_rk4[:, 0] - y_euler_interp))
    print(f"Maximum error between Euler (h={h_euler}) and RK4 (h={h_rk4}): {max_error:.6f}")

    # Calculate energy
    energy_rk4 = compute_energy(y_rk4[:, 0], y_rk4[:, 1], g, l)
    energy_euler = compute_energy(y_euler[:, 0], y_euler[:, 1], g, l)

    # Plotting solutions
    plt.figure(figsize=(10, 6))
    plt.plot(t_rk4, y_rk4[:, 0], label=f'RK4 (h={h_rk4})', color='blue', linestyle='-')
    plt.plot(t_euler, y_euler[:, 0], label=f'Euler (h={h_euler})', color='red', linestyle='--')
    plt.title("Comparison of Euler and RK4 Methods")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.xlim(0, 10)  # Restrict x-axis to (0, 10)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plotting energy
    plt.figure(figsize=(10, 6))
    plt.plot(t_rk4, energy_rk4, label="Total Energy (RK4)", color="blue", linestyle="-")
    plt.plot(t_euler, energy_euler, label="Total Energy (Euler)", color="red", linestyle="--")
    plt.title("Porównanie Energii Mechanicznej")
    plt.xlabel("Czas [s]")
    plt.ylabel("Energia całkowita [J]")
    plt.xlim(0, 10)  # Restrict x-axis to (0, 10)
    plt.ylim(2, 5)   # Restrict y-axis to (0, 6)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage: Change h_euler to test different time steps
h_euler = 0.0001  # Set the time step for Euler
compare_euler_with_rk4_and_energy(h_euler)