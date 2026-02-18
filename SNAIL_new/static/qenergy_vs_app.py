import sys
from pathlib import Path
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from helper.system import Hamiltonian

# Test parameters
Ej = 30.19
Ec = 0.1
omega_c1 = 5.226
omega_c2 = 8.135
phi_ex = 0.2
bare_dim = [10, 1, 6]
trunc_dim = [5, 1, 4]

sc = Hamiltonian(phi_ex, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)

# Function to calculate static rate for a given drive frequency
def calculate_static_rate(omegad, amplitude):
    der = sc.equasi_gradient(amplitude, omegad)[0]
    return np.abs(der)/2/np.pi


# Function to calculate analytical rate using exact omega_s
def calculate_analytical_rate2(omegad, amplitude):
    return np.abs(sc.der_formula2(amplitude, omegad))/2/np.pi


# First subplot - A = 0.5e-4*2*np.pi
A1 = 0.5e-4 * 2 * np.pi
omega_ds1 = np.linspace(6.158, 6.161, 200) * 2 * np.pi

# Parallelize the calculations using joblib for first amplitude
static_rates1 = Parallel(n_jobs=-1)(delayed(calculate_static_rate)(omegad, A1) for omegad in omega_ds1)
analytical_rates1_v2 = Parallel(n_jobs=-1)(delayed(calculate_analytical_rate2)(omegad, A1) for omegad in omega_ds1)
# Convert to numpy arrays for plotting
static_rates1 = np.array(static_rates1)
analytical_rates1_v2 = np.array(analytical_rates1_v2)

# Second subplot - A = 10e-3*2*np.pi
A2 = 10e-3 * 2 * np.pi
omega_ds2 = np.linspace(6.17, 6.23, 200) * 2 * np.pi

# Parallelize the calculations using joblib for second amplitude
static_rates2 = Parallel(n_jobs=-1)(delayed(calculate_static_rate)(omegad, A2) for omegad in omega_ds2)
analytical_rates2_v2 = Parallel(n_jobs=-1)(delayed(calculate_analytical_rate2)(omegad, A2) for omegad in omega_ds2)

# Convert to numpy arrays for plotting
static_rates2 = np.array(static_rates2)
analytical_rates2_v2 = np.array(analytical_rates2_v2)

# Create a figure with 1x2 subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Set font sizes for paper-quality figure
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 16,
    'mathtext.fontset': 'stix'
})

# Left: der_formula2 for A1
position = sc.state_index((1, 0, 0), sc.original_dim)
omega_s = sc.H_dressed[position, position].real / (2 * np.pi)
axs[0].plot(omega_ds1 / 2 / np.pi, static_rates1 * 1e3, label='Numerical', linewidth=2, color='blue', linestyle='-')
axs[0].plot(omega_ds1 / 2 / np.pi, analytical_rates1_v2 * 1e3, label='Analytical', linewidth=2, linestyle='--')
axs[0].set_xlabel(r'$\omega_d/2\pi$ (GHz)')
axs[0].set_ylabel(r'$\frac{\partial \Delta E_0}{\partial \Phi}/2\pi$ (GHz)')
axs[0].legend()
axs[0].tick_params(axis='both', labelsize=16)
axs[0].set_yscale('log')
axs[0].set_ylim(1e0, 1e3)
axs[0].set_title(r"$\Omega_0 = {:.2f}$ MHz".format(A1 * 1e3 / (2 * np.pi)))
axs[0].axvline(omega_s, color='black', linestyle='--', linewidth=1.5)
axs[0].text(
    omega_s,
    0.95,
    r'$\bar{\omega}_b = \omega_d$',
    transform=axs[0].get_xaxis_transform(),
    ha='right',
    va='top',
)

# Right: der_formula2 for A2
axs[1].plot(omega_ds2 / 2 / np.pi, static_rates2 * 1e3, label='Numerical', linewidth=2, color='blue', linestyle='-')
axs[1].plot(omega_ds2 / 2 / np.pi, analytical_rates2_v2 * 1e3, label='Analytical', linewidth=2, linestyle='--')
axs[1].set_xlabel(r'$\omega_d/2\pi$ (GHz)')
axs[1].set_ylabel(r'$\frac{\partial \Delta E_0}{\partial \Phi}/2\pi$ (GHz)')
axs[1].legend()
axs[1].tick_params(axis='both', labelsize=16)
axs[1].set_yscale('log')
axs[1].set_ylim(1e0, 1e3)
axs[1].set_title(r"$\Omega_0 = {:.1f}$ MHz".format(A2 * 1e3 / (2 * np.pi)))


plt.tight_layout()
plt.savefig('compare.pdf', bbox_inches='tight')
tex_dir = Path(__file__).resolve().parents[2] / '6949db0bfd19311f68336ca1' / 'Sec 2'
plt.savefig(tex_dir / 'compare.pdf', bbox_inches='tight')
plt.show()
