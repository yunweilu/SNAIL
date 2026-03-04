import numpy as np
import os
os.chdir('/home/yunwei/SNAIL/transmon/monte_carlo/new/full_simulation')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import qutip as qt
from hamiltonian_generator import Hamiltonian
from system import *
import pickle
import warnings


def fit_thermal_gamma_up(time_points, population_data, gamma_down=1/(2e4), N=1):
    """Fits N_2(t) = N * gamma_up/(gamma_up+gamma_down) * (1 - exp(-(gamma_up+gamma_down)*t)).
    Returns gamma_up."""
    def thermal_model(t, gamma_up):
        return N * gamma_up / (gamma_up + gamma_down) * (1 - np.exp(-(gamma_up + gamma_down) * t))
    try:
        popt, _ = curve_fit(thermal_model, time_points, population_data,
                            p0=[1e-6], bounds=(0, np.inf))
        return popt[0]
    except (RuntimeError, ValueError) as e:
        warnings.warn(f"Thermal gamma_up fit failed: {e}")
        return np.nan


DATA_FILE = 'thermal_debug_data.pkl'

with open(DATA_FILE, 'rb') as f:
    data = pickle.load(f)

A = data['A']
S0 = data['S0']
avg_rho = data['avg_rho']
time_points = data['time_points']

phi_ex = 0.2
Ej = 30.19
Ec = 0.1
sc = Hamiltonian(phi_ex, Ej, Ec, [5, 3])
optimal_omega, _ = sc.optimal_omegad(A)
optimal_omega = optimal_omega * 2 * np.pi
sc = Hamiltonian(phi_ex, Ej, Ec, [3, 2])
_, kick_and_sigmax, get_projector = sc.setup_floquet_system(A, optimal_omega)

P0 = np.zeros(len(time_points))
P2 = np.zeros(len(time_points))
for j, t in enumerate(time_points):
    fs, _, _ = kick_and_sigmax(t)
    projs = get_projector(fs)
    rho_j = qt.Qobj(avg_rho[j])
    P0[j] = np.real(qt.expect(projs[0], rho_j))
    P2[j] = np.real(qt.expect(projs[2], rho_j))

# --- Fit P[2] with thermal model ---
gamma_down = 1 / (2e4)
gamma_up = fit_thermal_gamma_up(time_points, P2, gamma_down=gamma_down, N=1)
print(f"P[2] thermal fit (γ↓={gamma_down:.2e}, N=1):")
print(f"  γ↑ = {gamma_up:.6e} /ns")
if not np.isnan(gamma_up):
    P2_ss = gamma_up / (gamma_up + gamma_down)
    print(f"  P[2] steady state = {P2_ss:.6e}")
    print(f"  γ↑ + γ↓ = {gamma_up + gamma_down:.6e} /ns")

# --- Generate fit curve ---
P2_fit = gamma_up / (gamma_up + gamma_down) * (1 - np.exp(-(gamma_up + gamma_down) * time_points))

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(time_points, P2, label='P[2] (data)')
ax.plot(time_points, P2_fit, '--', color='red',
        label=f'fit: $\\gamma_\\uparrow$={gamma_up:.2e} /ns')
ax.set_xlabel('Time (ns)', fontsize=14)
ax.set_ylabel('Population', fontsize=14)
ax.set_title(f'Ground state initial, A={A/(2*np.pi)*1e3:.2f} mHz, S0={S0:.1e}', fontsize=13)
ax.legend(fontsize=12)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('thermal_debug_P0_P2.pdf')
print("Plot saved to thermal_debug_P0_P2.pdf")
plt.show()
