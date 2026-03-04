import sys
sys.path.append('/home/yunwei/SNAIL/SNAIL_new')

import numpy as np
from system import *
from hamiltonian_generator import Hamiltonian as OldHamiltonian
from helper.system import Hamiltonian as NewHamiltonian

# ── Shared parameters ──
phi_ex = 0.2
Ej = 30.19
Ec = 0.1
A = 0.003e-3 * 2 * np.pi

# ── Old system (hamiltonian_generator.py) ──
print("=" * 60)
print("Old system (hamiltonian_generator.py)")
print("=" * 60)
sc_old = OldHamiltonian(phi_ex, Ej, Ec, [10, 6])
print(f"omega_s = {sc_old.omega_s / (2 * np.pi):.6f} GHz")

optimal_omega, min_rate = sc_old.optimal_omegad(A)
optimal_omega_rad = optimal_omega * 2 * np.pi

dr_old = sc_old.calculate_dr_exact(A, optimal_omega_rad)
print(f"calculate_dr_exact at optimal_omega = {dr_old:.6e}")

# ── New system (SNAIL_new/helper/system.py) ──
print()
print("=" * 60)
print("New system (SNAIL_new/helper/system.py)")
print("=" * 60)
bare_dim = [10, 1, 6]
trunc_dim = [5, 1, 4]
omega_c1 = 5.226
omega_c2 = 8.135

sc_new = NewHamiltonian(phi_ex, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)
position = sc_new.state_index((1, 0, 0), sc_new.original_dim)
omega_s_new = sc_new.H_dressed[position, position].real
print(f"omega_s = {omega_s_new / (2 * np.pi):.6f} GHz")

grad = sc_new.equasi_gradient(A, optimal_omega_rad)
rate_new = sc_new.static_rate(grad)
print(f"equasi_gradient at optimal_omega = {grad}")
print(f"static_rate from gradient = {rate_new}")

# ── Comparison ──
print()
print("=" * 60)
print("Comparison")
print("=" * 60)
print(f"Old dr_exact:  {dr_old:.6e}")
print(f"New static_rate (grad[0]): {rate_new[0]:.6e}")
