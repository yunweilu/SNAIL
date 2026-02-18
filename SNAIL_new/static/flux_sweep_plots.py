import sys
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

# Amplitude for quasi-energy calculation
A1 = 0.5e-4 * 2 * np.pi

# Sweep phi_ex around the nominal value
phi_sweep = np.linspace(phi_ex - 0.02, phi_ex + 0.02, 50)

# Set font sizes for paper-quality figure
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 14,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 14,
    'mathtext.fontset': 'stix'
})

# ============================================================
# Combined plot: Quasi-energy, omega_s vs phi_ex, and equasi_gradient vs omega_d
# ============================================================

# Three drive frequencies to test
omega_d_test = [6.163 * 2 * np.pi, 6.1597 * 2 * np.pi, 6.1587 * 2 * np.pi]
labels = ['$\\omega_d$ = 6.163 GHz', '$\\omega_d$ = 6.1597 GHz (peak)', '$\\omega_d$ = 6.1587 GHz (dip)']
colors = ['blue', 'red', 'green']

# Calculate quasi-energy vs phi for each omega_d
def calculate_quasienergy_vs_phi(phi_val, omegad, amplitude):
    sc_temp = Hamiltonian(phi_val, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)
    return sc_temp.quasi_energy(amplitude, omegad)[0]

# Calculate omega_s vs phi (using dressed basis)
def calculate_omegas_vs_phi(phi_val):
    sc_temp = Hamiltonian(phi_val, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)
    position = sc_temp.state_index((1, 0, 0), sc_temp.original_dim)
    return sc_temp.H_dressed[position, position]

# Create Hamiltonian for equasi_gradient calculation
sc = Hamiltonian(phi_ex, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)
omega_ds1 = np.linspace(6.157, 6.163, 50) * 2 * np.pi

def calculate_equasi_gradient(omegad, amplitude):
    return sc.equasi_gradient(amplitude, omegad)[0]

# Narrow phi sweep for first two plots
phi_sweep_narrow = np.linspace(phi_ex - 0.0002, phi_ex + 0.0002, 50)

# Create figure with 2x3 subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Top row: Quasi-energy vs phi_ex for each omega_d (separate plots)
for i, (omega_d, label, color) in enumerate(zip(omega_d_test, labels, colors)):
    print(f"Computing quasi-energy vs phi for {label}...")
    qe = Parallel(n_jobs=-1)(delayed(calculate_quasienergy_vs_phi)(p, omega_d, A1) for p in phi_sweep_narrow)
    qe = np.array(qe)
    axs[0, i].plot(phi_sweep_narrow, qe / 2 / np.pi, linewidth=2, color=color)
    axs[0, i].axvline(x=phi_ex, color='gray', linestyle='--', alpha=0.5)
    axs[0, i].set_xlabel('$\\phi_{ex}$ (flux quanta)')
    axs[0, i].set_ylabel('Quasi-energy (GHz)')
    axs[0, i].set_title(f'{label}')

# Bottom left: omega_s vs phi_ex (narrow range)
print("Computing omega_s vs phi...")
omegas_sweep = Parallel(n_jobs=-1)(delayed(calculate_omegas_vs_phi)(p) for p in phi_sweep_narrow)
omegas_sweep = np.array(omegas_sweep)

axs[1, 0].plot(phi_sweep_narrow, omegas_sweep / 2 / np.pi, linewidth=2, color='blue')
axs[1, 0].axvline(x=phi_ex, color='gray', linestyle='--', alpha=0.5, label=f'$\\phi_{{ex}}$ = {phi_ex}')
axs[1, 0].set_xlabel('$\\phi_{ex}$ (flux quanta)')
axs[1, 0].set_ylabel('$\\omega_s$ (GHz)')
axs[1, 0].set_title('SNAIL frequency $\\omega_s$ vs flux')
axs[1, 0].legend()

# Bottom middle: equasi_gradient vs drive frequency
print("Computing equasi_gradient vs drive frequency...")
equasi_grads = Parallel(n_jobs=-1)(delayed(calculate_equasi_gradient)(omegad, A1) for omegad in omega_ds1)
equasi_grads = np.array(equasi_grads)

axs[1, 1].plot(omega_ds1 / 2 / np.pi, np.abs(equasi_grads) / 2 / np.pi, linewidth=2, color='blue')
axs[1, 1].set_xlabel('$\\omega_d$ (GHz)')
axs[1, 1].set_ylabel('$|\\partial \\varepsilon / \\partial \\phi_{ex}|$ (GHz)')
axs[1, 1].set_title(f'Quasi-energy gradient vs $\\omega_d$, $A$ = {A1 * 1e3 / (2 * np.pi):.1f} MHz')
axs[1, 1].set_yscale('log')

# Bottom right: empty or can be used for another plot
axs[1, 2].axis('off')

plt.tight_layout()
plt.savefig('flux_sweep_combined.pdf', bbox_inches='tight')
plt.show()
