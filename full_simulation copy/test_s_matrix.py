import numpy as np
from joblib import Parallel, delayed
from hamiltonian_generator import Hamiltonian

phi_ex = 0.2
Ej = 30.19
Ec = 0.1

A_values = [1e-4*2*np.pi]

sc = Hamiltonian(phi_ex, Ej, Ec, [5, 3])
print(f"omega_s = {sc.omega_s / (2 * np.pi):.6f} GHz")

def process_A(A):
    sc_big = Hamiltonian(phi_ex, Ej, Ec, [10, 6])
    optimal_omega, min_rate = sc_big.optimal_omegad(A)
    optimal_omega_rad = optimal_omega * 2 * np.pi

    sc_small = Hamiltonian(phi_ex, Ej, Ec, [5, 3])
    noise, H, H_control, s = sc_small.calculate_floquet_U_test(A, optimal_omega_rad)

    detuning = optimal_omega_rad - sc_small.omega_s
    s_diff = s[1, 1] - s[0, 0]
    rate = np.abs(s_diff) ** 2 / 2 / 2e4
    sds = s.conj().T @ s

    return {
        'A': A,
        'optimal_omega': optimal_omega,
        'min_rate': min_rate,
        'detuning': detuning,
        's00': s[0, 0],
        's11': s[1, 1],
        's_diff': s_diff,
        'A_over_detuning': A / detuning,
        'rate': rate,
        's': s,
        'sds': sds,
    }

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(process_A)(A) for A in A_values
)

print(f"\n{'='*80}")
print(f"{'A (mHz)':>10} | {'optimal_wd (GHz)':>16} | {'min_rate':>12} | {'A/detuning':>12} | {'s[0,0]':>16} | {'s[1,1]':>16} | {'s[1,1]-s[0,0]':>16} | {'|.|^2/2/2e4':>14}")
print(f"{'-'*130}")
for r in results:
    print(f"{r['A']/(2*np.pi)*1e3:>10.4f} | {r['optimal_omega']:>16.6f} | {r['min_rate']:>12.6e} | {r['A_over_detuning']:>12.6f} | {r['s00']:>16.6e} | {r['s11']:>16.6e} | {r['s_diff']:>16.6e} | {r['rate']:>14.6e}")

    print(f"\n  Transformed s matrix:")
    print(np.array2string(r['s'], precision=6, suppress_small=True))

    print(f"\n  s^dagger @ s:")
    print(np.array2string(r['sds'], precision=6, suppress_small=True))
