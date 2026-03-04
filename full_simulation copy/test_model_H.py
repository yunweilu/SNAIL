import numpy as np
from system import sort_eigenpairs
from hamiltonian_generator import Hamiltonian

phi_ex = 0.2
Ej = 30.19
Ec = 0.1
trunc = [5, 3]

sc = Hamiltonian(phi_ex, Ej, Ec, trunc)

omega_s = sc.omega_s
omega_c = sc.omega_c
chi = sc.chi
K = sc.anh
phi_zpf = sc.phi_zpf
s_op = sc.s

print(f"omega_s = {omega_s / (2*np.pi):.6f} GHz")
print(f"omega_c = {omega_c / (2*np.pi):.6f} GHz")
print(f"chi     = {chi / (2*np.pi)*1e3:.4f} MHz")
print(f"K (anh) = {K / (2*np.pi)*1e3:.4f} MHz")
phi_zpf = float(phi_zpf)
print(f"phi_zpf = {phi_zpf:.6f}")

N_s, N_c = trunc
dim = N_s * N_c

# SNAIL operators (b)
b_s = np.diag(np.sqrt(np.arange(1, N_s)), 1)
bd_s = b_s.T
nb_s = bd_s @ b_s
I_s = np.eye(N_s)

# Cavity operators (c)
a_c = np.diag(np.sqrt(np.arange(1, N_c)), 1)
ad_c = a_c.T
nc_c = ad_c @ a_c
I_c = np.eye(N_c)

# Full space operators
nb = np.kron(nb_s, I_c)
nc = np.kron(I_s, nc_c)
b = np.kron(b_s, I_c)
bd = np.kron(bd_s, I_c)

A_values = np.linspace(0.003e-3, 0.03e-3, 20) * 2 * np.pi

from joblib import Parallel, delayed

def process_A(A, omega_s, omega_c, chi, K, phi_zpf, s_op, nb, nc, b, bd):
    sc_big = Hamiltonian(phi_ex, Ej, Ec, [10, 6])
    optimal_omega, min_rate = sc_big.optimal_omegad(A)
    omega_d = optimal_omega * 2 * np.pi
    Delta_bd = omega_s - omega_d

    H_R = (Delta_bd * nb
           + (K / 2) * bd @ bd @ b @ b
           + omega_c * nc
           + chi * nb @ nc
           + A / (2 * phi_zpf)/2 * (bd + b))

    evals, U = np.linalg.eigh(H_R)
    evals, U = sort_eigenpairs(evals, U)
    Ud = U.T.conj()

    s_transformed = Ud @ s_op @ U

    s_diff = s_transformed[1, 1] - s_transformed[0, 0]
    detuning = omega_d - omega_s
    rate = np.abs(s_diff) ** 2 / 2 / 2e4

    return {
        'A': A,
        'omega_d': omega_d,
        'detuning': detuning,
        's_diff': s_diff,
        'A_over_detuning': A / detuning,
        'rate': rate,
    }

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(process_A)(A, omega_s, omega_c, chi, K, phi_zpf, s_op, nb, nc, b, bd)
    for A in A_values
)

print(f"\n{'='*90}")
print(f"{'A (mHz)':>10} | {'omega_d (GHz)':>14} | {'detuning (MHz)':>14} | {'A/detuning':>12} | {'s[1,1]-s[0,0]':>16} | {'|.|^2/2/2e4':>14}")
print(f"{'-'*90}")
for r in results:
    print(f"{r['A']/(2*np.pi)*1e3:>10.4f} | {r['omega_d']/(2*np.pi):>14.6f} | {r['detuning']/(2*np.pi)*1e3:>14.4f} | {r['A_over_detuning']:>12.6f} | {r['s_diff']:>16.6e} | {r['rate']:>14.6e}")
