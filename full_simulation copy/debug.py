import numpy as np
import qutip as qt
from scipy.linalg import logm
from system import sort_eigenpairs
from hamiltonian_generator import Hamiltonian

phi_ex = 0.2
Ej = 30.19
Ec = 0.1

A = 0.003e-3 * 2 * np.pi

sc_big = Hamiltonian(phi_ex, Ej, Ec, [10, 6])
optimal_omega, min_rate = sc_big.optimal_omegad(A)
optimal_omega_rad = optimal_omega * 2 * np.pi

sc = Hamiltonian(phi_ex, Ej, Ec, [5, 3])
print(f"omega_s = {sc.omega_s / (2 * np.pi):.6f} GHz")
print(f"optimal_omega_d = {optimal_omega:.6f} GHz")
print(f"A = {A / (2 * np.pi) * 1e3:.4f} mHz")

# Replicate calculate_floquet_U_test but print intermediate results
H0 = qt.Qobj(sc.H)
Hc = qt.Qobj(sc.H_control)
T = (2 * np.pi) / optimal_omega_rad

H = [H0, [Hc, lambda t, args: A * np.cos(args['w'] * t)]]
floquet_basis = qt.FloquetBasis(H, T, args={'w': optimal_omega_rad})

f_modes = floquet_basis.mode(0)
f_energies = floquet_basis.e_quasi

print(f"\nFloquet quasi-energies (raw):")
print(f_energies)

f_modes_array = []
for mode in f_modes:
    f_modes_array.append(mode.full().flatten())
f_modes = np.array(f_modes_array).T

print(f"\nFloquet modes matrix shape: {f_modes.shape}")

evals, U = sort_eigenpairs(f_energies, f_modes)
Ud = U.T.conj()

print(f"\nSorted Floquet quasi-energies:")
print(evals)
print(f"\nFloquet unitary U (absolute values):")
print(np.abs(U))

print(f"\nFloquet unitary U:")
print(U)

H_gen = 1j * logm(U) 
print(f"\nGenerator of Floquet unitary (H = i*logm(U)/T):")
print(H_gen)
print(f"\nGenerator diagonal (real parts):")
print(np.real(np.diag(H_gen)))

N_s, N_c = 2, 2
idx_g1 = 0 * N_c + 1  # |g1> = (n=0, k=1) -> index 1
idx_e1 = 1 * N_c + 1  # |e1> = (n=1, k=1) -> index 4
print(f"\n|g1> index = {idx_g1}, |e1> index = {idx_e1}")
print(f"U[e1, g1] = U[{idx_e1}, {idx_g1}] = {U[idx_e1, idx_g1]}")
print(f"|U[e1, g1]| = {np.abs(U[idx_e1, idx_g1]):.6e}")

s = Ud @ sc.s @ U
sds = sc.s.conj().T @ sc.s
sds_floquet = Ud @ sds @ U

print(f"\ns†s in Floquet basis (absolute values):")
print(np.abs(sds_floquet[:6, :6]))

print(f"\n(s†s)[1,1] - (s†s)[0,0] = {sds_floquet[1, 1] - sds_floquet[0, 0]}")

print(f"\ns matrix in Floquet basis (absolute values):")
print(np.abs(s[:6, :6]))

print(f"\ns[1,1] - s[0,0] = {s[1, 1] - s[0, 0]}")

detuning = optimal_omega_rad - sc.omega_s
print(f"\nA/detuning = {A / detuning:.6f}")
print(f"abs(s[1,1]-s[0,0])**2/2/2e4 = {np.abs(s[1, 1] - s[0, 0])**2 / 2 / 2e4:.6e}")
