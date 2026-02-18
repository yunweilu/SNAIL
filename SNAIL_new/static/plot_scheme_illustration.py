import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from helper.noise_generator import GenerateNoise

sample_rate = 1               # per ns
tnoise_max = int(1e6)             # number of samples
omega_ir = 1/tnoise_max*2*np.pi
S0 = 1e-6
relative_PSD_strength = S0**2
num_realizations = 100
ifwhite = False               # True for white noise, False for 1/f noise

# Generate white noise (unit variance) trajectories
N = tnoise_max * sample_rate
gn = GenerateNoise(sample_rate, tnoise_max, relative_PSD_strength, num_realizations, ifwhite)
trajs = gn.generate_colored_noise()

# Time array
t = np.arange(tnoise_max) / sample_rate  # time in ns

# ============================================================
# Plot cavity frequency vs time with flux noise
# ============================================================
from helper.system import Hamiltonian
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar

# Parameters from qenergy_vs_app
Ej = 30.19
Ec = 0.1
omega_c1 = 5.226
omega_c2 = 8.135
phi_ex = 0.2
bare_dim = [10, 1, 6]
trunc_dim = [5, 1, 4]

# Take first 200 points of noise trajectory
n_points = 200
noise_traj = trajs[0][:n_points]
t_short = t[:n_points]

# Calculate cavity frequency for each noisy phi_ex value
def get_cavity_freq(phi_val):
    sc = Hamiltonian(phi_val, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)
    # Get cavity frequency from dressed Hamiltonian (state |0,0,1>)
    position = sc.state_index((0, 0, 1), sc.original_dim)
    return np.real(sc.H_dressed[position, position]) / 2 / np.pi  # in GHz

# Compute cavity frequency vs time (parallelized)
phi_noisy = phi_ex + noise_traj
cavity_freqs = np.array(Parallel(n_jobs=-1)(delayed(get_cavity_freq)(p) for p in phi_noisy))

# ============================================================
# Find optimal omega_d that minimizes quasi-energy gradient (sweet spot)
# ============================================================

A = 10e-3 * 2 * np.pi  # 10 MHz
omega_d_init = 6.188 * 2 * np.pi

# Create Hamiltonian at nominal phi_ex for optimization
sc_nom = Hamiltonian(phi_ex, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)

def objective(omega_d_val):
    """Minimize the absolute gradient of quasi-energy w.r.t. flux."""
    grad = sc_nom.equasi_gradient(A, omega_d_val)[0]
    return np.abs(grad)

# Optimize omega_d to find sweet spot
print("Optimizing omega_d to find sweet spot...")
result = minimize_scalar(objective, bounds=(6.15 * 2 * np.pi, 6.22 * 2 * np.pi), method='bounded')
omega_d_opt = result.x
print(f"Optimal omega_d = {omega_d_opt / 2 / np.pi:.6f} GHz")

# ============================================================
# Plot quasi-energy vs time at optimal omega_d
# ============================================================

# Calculate quasi-energy for each noisy phi_ex value at optimal omega_d
def get_quasi_energy_opt(phi_val):
    sc = Hamiltonian(phi_val, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)
    return sc.quasi_energy(A, omega_d_opt)[0] / 2 / np.pi  # in GHz

# Compute quasi-energy vs time (parallelized)
print("Computing quasi-energy vs time at optimal omega_d...")
quasi_energies_opt = np.array(Parallel(n_jobs=-1)(delayed(get_quasi_energy_opt)(p) for p in phi_noisy))

# ============================================================
# Combined plot: noise background, undriven (left y), driven (right y)
# ============================================================

fig_combined, ax_left = plt.subplots(1, 1, figsize=(12, 5))

# Plot noise as grey line (scaled to fit cavity frequency range)
noise_scaled = cavity_freqs.min() + (noise_traj - noise_traj.min()) / (noise_traj.max() - noise_traj.min()) * (cavity_freqs.max() - cavity_freqs.min())
ax_left.plot(t_short, noise_scaled, linewidth=1, color='grey', alpha=0.5)

# Left y-axis: Undriven cavity frequency
ax_left.plot(t_short, cavity_freqs, linewidth=1.5, color='black', label='Undriven (cavity freq)')
ax_left.set_xlabel('Time (ns)')
ax_left.set_ylabel('Cavity frequency (GHz)', color='black')
ax_left.tick_params(axis='y', labelcolor='black')

# Right y-axis: Driven quasi-energy
ax_right = ax_left.twinx()
ax_right.plot(t_short, quasi_energies_opt, linewidth=1.5, color='red', label=f'Driven ($\\omega_d$={omega_d_opt/2/np.pi:.4f} GHz)')
ax_right.set_ylabel('Quasi-energy (GHz)', color='red')
ax_right.tick_params(axis='y', labelcolor='red')

# Set same y-axis range for both axes to make fluctuation comparison intuitive
left_range = cavity_freqs.max() - cavity_freqs.min()
right_range = quasi_energies_opt.max() - quasi_energies_opt.min()
max_range = max(left_range, right_range)

left_center = (cavity_freqs.max() + cavity_freqs.min()) / 2
right_center = (quasi_energies_opt.max() + quasi_energies_opt.min()) / 2

ax_left.set_ylim(left_center - max_range/2 * 1.1, left_center + max_range/2 * 1.1)
ax_right.set_ylim(right_center - max_range/2 * 1.1, right_center + max_range/2 * 1.1)

# Make both y-axes have consistent number of ticks
from matplotlib.ticker import MaxNLocator
ax_left.yaxis.set_major_locator(MaxNLocator(nbins=8))
ax_right.yaxis.set_major_locator(MaxNLocator(nbins=8))

# Title
ax_left.set_title(f'Frequency fluctuation: Undriven vs Driven ($A$ = 10 MHz)')

# Add text labels near each line (no legend box)
# Driven: close to the rightmost point
ax_right.text(t_short[-1], quasi_energies_opt[-1], '  Driven', color='red', fontsize=11, fontweight='bold', va='center', ha='left')
# Undriven: close to maximum point
idx_max = np.argmax(cavity_freqs)
ax_left.text(t_short[idx_max], cavity_freqs[idx_max], '  Undriven', color='black', fontsize=11, fontweight='bold', va='bottom', ha='left')
# Flux fluctuation: close to minimum point
idx_min = np.argmin(noise_scaled)
ax_left.text(t_short[idx_min], noise_scaled[idx_min], '  Flux fluctuation', color='grey', fontsize=11, fontweight='bold', va='top', ha='left')

plt.tight_layout()
plt.savefig('combined_freq_comparison.pdf', bbox_inches='tight')
plt.show()
