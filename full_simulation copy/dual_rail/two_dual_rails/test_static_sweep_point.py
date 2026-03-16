import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path("/Users/yunwei/Desktop/project/cavity dephasing/SNAIL/full_simulation copy/dual_rail/two_dual_rails")))
from system import Hamiltonian

Ej = 30.19
Ec = 0.1
omega_c1 = 5.226
omega_c2 = 7.335
phi_ex = 0.2
bare_dim = [10, 6, 6]
trunc_dim = [5, 2, 2]
g_val = 0.05 * 2 * np.pi

def build(phi):
    sc = Hamiltonian(phi, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)
    sc.g = g_val
    sc.H, sc.H_control, sc.H_flux_drive = sc.get_H()
    sc.H_dressed, sc.H_control_dressed, sc.H_flux_drive_dressed = sc.dressed_basis()
    return sc

sc0 = build(phi_ex)
dim = sc0.original_dim
idx_0 = sc0.state_index((0, 0, 0), dim)
idx_b = sc0.state_index((1, 0, 0), dim)

omega_q = sc0.H_dressed[idx_b, idx_b].real - sc0.H_dressed[idx_0, idx_0].real

detuning_mhz = -28.4
optimal_omega = omega_q + (detuning_mhz * 1e-3) * 2 * np.pi
A = 10e-3 * 2 * np.pi

delta_phi = 1e-6
sc_p = build(phi_ex + delta_phi)
sc_m = build(phi_ex - delta_phi)

E_c1_p, E_c2_p = sc_p.quasi_energy(A, optimal_omega)
E_c1_m, E_c2_m = sc_m.quasi_energy(A, optimal_omega)

grad_c1 = (E_c1_p - E_c1_m) / (2 * delta_phi) / (2 * np.pi)
grad_c2 = (E_c2_p - E_c2_m) / (2 * delta_phi) / (2 * np.pi)

print(f"Sweep Point -28.4 MHz Result -> Cavity 1: {grad_c1}, Cavity 2: {grad_c2}")
