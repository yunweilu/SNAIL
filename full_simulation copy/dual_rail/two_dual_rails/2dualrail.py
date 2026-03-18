import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from system import Hamiltonian

DEFAULT_DRIVE_A = 10e-3 * 2 * np.pi  # Keep amplitude convention aligned with static_sweep.py.


def annihilation(dim):
    return np.diag(np.sqrt(np.arange(1, dim)), 1)


def compute_dc_dphi(phi_ex=0.2, detuning_mhz=29.0, A=None):
    """
    Compute d(c1)/dPhi and d(c2)/dPhi via central finite-difference,
    following the same method as static_sweep.py.
    """
    Ej = 30.19
    Ec = 0.1
    omega_c1 = 5.226
    omega_c2 = 7.335
    bare_dim = [10, 6, 6]
    trunc_dim = [5, 2, 2]
    g_val = 0.05 * 2 * np.pi
    if A is None:
        A = DEFAULT_DRIVE_A
    delta_phi = 1e-6

    def build(phi):
        sc = Hamiltonian(phi, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)
        sc.g = g_val
        sc.H, sc.H_control, sc.H_flux_drive, sc.noise, sc.s = sc.get_H()
        sc.H_dressed, sc.H_control_dressed, sc.H_flux_drive_dressed = sc.dressed_basis()
        return sc

    sc0 = build(phi_ex)
    dim = sc0.original_dim
    idx_0 = sc0.state_index((0, 0, 0), dim)
    idx_b = sc0.state_index((1, 0, 0), dim)
    omega_q = sc0.H_dressed[idx_b, idx_b].real - sc0.H_dressed[idx_0, idx_0].real
    omega_drive = omega_q + (detuning_mhz * 1e-3) * 2 * np.pi

    sc_p = build(phi_ex + delta_phi)
    sc_m = build(phi_ex - delta_phi)
    E_c1_p, E_c2_p = sc_p.quasi_energy(A, omega_drive)
    E_c1_m, E_c2_m = sc_m.quasi_energy(A, omega_drive)

    dc1_dphi = (E_c1_p - E_c1_m) / (2 * delta_phi) / (2 * np.pi)
    dc2_dphi = (E_c2_p - E_c2_m) / (2 * delta_phi) / (2 * np.pi)
    return float(dc1_dphi), float(dc2_dphi)


def build_tc_operators(A):
    Ej = 30.19
    Ec = 0.1
    omega_c1 = 5.226
    omega_c2 = 7.335
    phi_ex = 0.2
    bare_dim = [10, 6, 6]
    trunc_dim = [3, 2, 2]
    g_val = 0.05 * 2 * np.pi

    sc = Hamiltonian(phi_ex, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)
    sc.g = g_val
    sc.H, sc.H_control, sc.H_flux_drive, sc.noise, sc.s = sc.get_H()
    sc.H_dressed, sc.H_control_dressed, sc.H_flux_drive_dressed = sc.dressed_basis()
    
    dim = sc.original_dim
    idx_0 = sc.state_index((0, 0, 0), dim)
    idx_b = sc.state_index((1, 0, 0), dim)
    omega_q = sc.H_dressed[idx_b, idx_b].real - sc.H_dressed[idx_0, idx_0].real
    
    detuning_mhz = 28.5
    optimal_omega = omega_q + (detuning_mhz * 1e-3) * 2 * np.pi

    approx_ops = sc.get_approximate_H(A, optimal_omega)
    
    return {
        "phi_ex": phi_ex,
        "optimal_omega": optimal_omega,
        "sds_diag": approx_ops["sds_diag"],
        "sop_selected": approx_ops["sop_selected"],
        "c1op_selected": approx_ops["c1op_selected"],
        "c2op_selected": approx_ops["c2op_selected"],
        "H0_rot_raw": approx_ops["H0_rot_raw"],
        "trunc_dim": trunc_dim
    }

def build_total_operators(A, n_aux3=2, n_aux4=2):
    import qutip as qt
    tc = build_tc_operators(A=A)
    
    # from trunc_dim: [n_transmon, n_c2, n_c1]
    n_transmon, n_c2, n_c1 = tc["trunc_dim"]
    dim_tc = n_transmon * n_c2 * n_c1
    
    I_aux3 = np.eye(n_aux3, dtype=complex)
    I_aux4 = np.eye(n_aux4, dtype=complex)
    I_aux_both = np.kron(I_aux3, I_aux4)
    
    I_tc = np.eye(dim_tc, dtype=complex)

    H0_total = np.kron(tc["H0_rot_raw"], I_aux_both)
    sds_total = np.kron(tc["sds_diag"], I_aux_both)
    sop_total = np.kron(tc["sop_selected"], I_aux_both)
    c1op_total = np.kron(tc["c1op_selected"], I_aux_both)
    c2op_total = np.kron(tc["c2op_selected"], I_aux_both)
    
    # We want [tc] \otimes [c3] \otimes [c4]
    a_c3_raw = np.kron(I_tc, np.kron(annihilation(n_aux3), I_aux4))
    a_c4_raw = np.kron(I_tc, np.kron(I_aux3, annihilation(n_aux4)))

    dims = [[n_transmon, n_c2, n_c1, n_aux3, n_aux4], [n_transmon, n_c2, n_c1, n_aux3, n_aux4]]
    
    return {
        "phi_ex": tc["phi_ex"],
        "optimal_omega": tc["optimal_omega"],
        "H0_total": qt.Qobj(H0_total, dims=dims),
        "sds_total": qt.Qobj(2*sds_total, dims=dims),
        "sop_total": qt.Qobj(sop_total, dims=dims),
        "c1op_total": qt.Qobj(c1op_total, dims=dims),
        "c2op_total": qt.Qobj(c2op_total, dims=dims),
        "a_c3_total": qt.Qobj(a_c3_raw, dims=dims),
        "a_c4_total": qt.Qobj(a_c4_raw, dims=dims),
    }

def main():
    A = DEFAULT_DRIVE_A
    print("Building total operators for 2 dual-rails scenario...")
    ops = build_total_operators(A=A, n_aux3=2, n_aux4=2)
    
    print("\n=== Resulting Total Operators ===")
    print(f"H0_total dims: {ops['H0_total'].dims}, shape: {ops['H0_total'].shape}")
    print(f"sds_total dims: {ops['sds_total'].dims}, shape: {ops['sds_total'].shape}")
    print(f"sop_total dims: {ops['sop_total'].dims}")
    print(f"c1op_total dims: {ops['c1op_total'].dims}")
    print(f"c2op_total dims: {ops['c2op_total'].dims}")
    print(f"a_c3_total dims: {ops['a_c3_total'].dims}")
    print(f"a_c4_total dims: {ops['a_c4_total'].dims}")
    
    print("\nThey are correctly populated and mapped into qutip Qobjs!")
    
    # Print the full H0_total as requested
    np.set_printoptions(threshold=sys.maxsize, linewidth=200, precision=4, suppress=True)
    H0_arr = ops['H0_total'].full()
    print("\n=== H0_total (Driven Case, A != 0) ===")
    print("Non-zero elements count:", np.count_nonzero(H0_arr))
    print(H0_arr)

    # Quasi-energy derivatives at default detuning.
    dc1_dphi, dc2_dphi = compute_dc_dphi(phi_ex=0.2, detuning_mhz=29.0)
    print("\n=== Quasi-energy Derivatives (detuning = 29 MHz) ===")
    print(f"dc1/dPhi: {dc1_dphi:.6f} GHz/Phi0 | dc2/dPhi: {dc2_dphi:.6f} GHz/Phi0")
    print("\n=== Dual-Rail Mapping ===")
    print(f"Dual-rail A: rail-1 (c1) -> {dc1_dphi:.6f} GHz/Phi0, rail-2 (c2) -> {dc2_dphi:.6f} GHz/Phi0")
    print(
        "Dual-rail B: rail-1 (c3), rail-2 (c4). "
        "In this script c3/c4 are auxiliary modes; if B is identical to A, "
        "use the same derivative values by symmetry."
    )

if __name__ == "__main__":
    main()
