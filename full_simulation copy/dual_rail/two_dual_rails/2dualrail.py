import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from system import Hamiltonian


def annihilation(dim):
    return np.diag(np.sqrt(np.arange(1, dim)), 1)

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
    
    detuning_mhz = 28.4
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
    A = 10e-3 * 2 * np.pi
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

if __name__ == "__main__":
    main()
