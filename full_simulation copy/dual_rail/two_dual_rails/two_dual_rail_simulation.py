import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from qutip import matrix_histogram
from joblib import Parallel, delayed

import sys
sys.path.insert(0, "/Users/yunwei/Desktop/project/cavity dephasing/SNAIL/full_simulation copy/dual_rail/single_dual_rial")
from noise_generator import GenerateNoise

sys.path.insert(0, "/Users/yunwei/Desktop/project/cavity dephasing/SNAIL/full_simulation copy/dual_rail/two_dual_rails")
from importlib.machinery import SourceFileLoader
twodualrail = SourceFileLoader("2dualrail", "/Users/yunwei/Desktop/project/cavity dephasing/SNAIL/full_simulation copy/dual_rail/two_dual_rails/2dualrail.py").load_module()


def basis_index(t, c2, c1, c3, c4, trunc_dim):
    n_t, n_c2, n_c1 = trunc_dim
    n_c3 = 2
    n_c4 = 2
    return t * (n_c2 * n_c1 * n_c3 * n_c4) + c2 * (n_c1 * n_c3 * n_c4) + c1 * (n_c3 * n_c4) + c3 * n_c4 + c4

def build_dressed_subspace_from_h0(H0_total, trunc_dim):
    H_arr = np.asarray(H0_total.full(), dtype=complex)
    evals, U = np.linalg.eigh(H_arr)
    
    n_t, n_c2, n_c1 = trunc_dim
    
    sorted_indices = []
    for i in range(U.shape[1]):
        max_abs_vals = np.abs(U[i, :])
        max_index = np.argmax(max_abs_vals)
        while max_index in sorted_indices:
            max_abs_vals[max_index] = -np.inf
            max_index = np.argmax(max_abs_vals)
        sorted_indices.append(max_index)
    U = U[:, sorted_indices]
    evals = evals[sorted_indices]

    label_to_index = {
        "00_LL": basis_index(0, 1, 1, 0, 0, trunc_dim),
        "01_LL": basis_index(0, 0, 1, 0, 1, trunc_dim),
        "10_LL": basis_index(0, 1, 0, 1, 0, trunc_dim),
        "11_LL": basis_index(0, 0, 0, 1, 1, trunc_dim),
    }

    dims_ket = [[n_t, n_c2, n_c1, 2, 2], [1, 1, 1, 1, 1]]
    dressed_kets = {}
    dressed_projectors = {}
    for lbl, idx in label_to_index.items():
        if idx >= U.shape[1]:
            continue
        ket = qt.Qobj(U[:, idx], dims=dims_ket)
        dressed_kets[lbl] = ket
        dressed_projectors[lbl] = ket * ket.dag()

    return {
        "evals": evals,
        "U": U,
        "dressed_kets": dressed_kets,
        "dressed_projectors": dressed_projectors,
        "labels": ["00_LL", "01_LL", "10_LL", "11_LL"],
    }

def build_initial_state(trunc_dim):
    n_t, n_c2, n_c1 = trunc_dim
    n_c3 = 2
    n_c4 = 2
    
    ket00 = qt.tensor(qt.basis(n_t, 0), qt.basis(n_c2, 1), qt.basis(n_c1, 1), qt.basis(n_c3, 0), qt.basis(n_c4, 0))
    ket10 = qt.tensor(qt.basis(n_t, 0), qt.basis(n_c2, 1), qt.basis(n_c1, 0), qt.basis(n_c3, 1), qt.basis(n_c4, 0))
    
    return (ket00 + ket10).unit()

def generate_noise_trajs(noise_t_max, noise_dt, num_realizations, S0):
    n_noise_samples = int(np.ceil(float(noise_t_max) / float(noise_dt)))
    gn = GenerateNoise(
        1,
        n_noise_samples,
        S0**2,
        num_realizations,
        False,
    )
    return gn.generate_colored_noise()

def simulate_single_trajectory(traj_idx, raw_trajs, noise_dt, solver_nsteps, sds_total, H0_total, psi0, time_points, c_ops):
    opts = {
        "nsteps": int(solver_nsteps),
        "atol": 1e-12,
        "rtol": 1e-12,
        "progress_bar": False,
    }
    traj = np.asarray(raw_trajs[traj_idx], dtype=complex)
    t_noise = np.arange(len(traj), dtype=float) * noise_dt
    noise_qevo = qt.coefficient(np.asarray(traj, dtype=complex), tlist=t_noise)
    H = qt.QobjEvo([H0_total, [1.5*sds_total, noise_qevo]])

    result = qt.mesolve(H, psi0, time_points, c_ops, options=opts)
    rho_t = [st if st.isoper else qt.ket2dm(st) for st in result.states]
    return np.stack([rho.full() for rho in rho_t], axis=0)

def logical_density_from_full_rho(rho_full, kets):
    rho_logical_arr = np.zeros((4, 4), dtype=complex)
    for i, lbl_i in enumerate(["00_LL", "01_LL", "10_LL", "11_LL"]):
        for j, lbl_j in enumerate(["00_LL", "01_LL", "10_LL", "11_LL"]):
            if lbl_i in kets and lbl_j in kets:
                var1 = kets[lbl_i]
                var2 = rho_full * kets[lbl_j]
                val = var1.dag() * var2
                
                # Check if it was returned as a complex scalar or a Qobj
                from numbers import Number
                if isinstance(val, Number):
                    rho_logical_arr[i, j] = complex(val)
                elif hasattr(val, "tr"):
                    # Typically a Qobj trace will give you the inner product if dims align 
                    # but depending on dims Qobj can hold [[val]]
                    rho_logical_arr[i, j] = complex(val.full()[0, 0] if val.shape == (1,1) else val.tr())
                else: 
                     rho_logical_arr[i, j] = complex(val)
                     
    return qt.Qobj(rho_logical_arr, dims=[[4], [4]])

def perform_tomography_and_plot(rho_full, kets, title, filename):
    rho_logical = logical_density_from_full_rho(rho_full, kets)
    
    # Trace over internal degrees of freedom
    rho_logical = rho_logical / rho_logical.tr()
    
    labels = ["|00>", "|01>", "|10>", "|11>"]
    fig, ax = matrix_histogram(rho_logical, labels, labels)
    
    ax.set_title(title)
    ax.view_init(azim=-55, elev=45)
    plt.tight_layout()
    plot_out = Path(filename).resolve()
    fig.savefig(plot_out, dpi=200, bbox_inches="tight")
    print(f"Saved logical tomography plot to: {plot_out}")

def main():
    parser = argparse.ArgumentParser(description="Two Dual-rails dynamics")
    parser.add_argument("--A-over-pi", type=float, default=10e-3, help="Drive amplitude")
    parser.add_argument("--sim-time", type=float, default=60000.0, help="Simulation end time.")
    parser.add_argument("--num-time-points", type=int, default=10, help="Number of saved density-matrix points.")
    parser.add_argument("--num-realizations", type=int, default=20, help="Number of noise trajectories.")
    parser.add_argument("--S0", type=float, default=1e-5, help="Noise amplitude.")
    parser.add_argument("--noise-t-max", type=float, default=int(1e8), help="Tracking max time")
    parser.add_argument("--noise-dt", type=float, default=1000.0, help="Tracking dt")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel processing.")
    parser.add_argument("--gamma-main", type=float, default=1 / (2e4), help="Decay main")
    parser.add_argument("--gamma-aux", type=float, default=5e-7, help="Decay aux")
    args = parser.parse_args()

    ops_undriven = twodualrail.build_total_operators(A=0.0)
    driven_A = args.A_over_pi * np.pi
    ops_driven = twodualrail.build_total_operators(A=driven_A)
    
    trunc_dim = ops_undriven.get("trunc_dim", [3, 2, 2])
    
    psi0 = build_initial_state(trunc_dim)
    dressed_undriven = build_dressed_subspace_from_h0(ops_undriven["H0_total"], trunc_dim)
    
    print("Plotting Initial State Tomography...")
    perform_tomography_and_plot(qt.ket2dm(psi0), dressed_undriven["dressed_kets"], "Initial State Tomography (t=0)", "tomo_initial.png")

    time_points = np.linspace(0.0, args.sim_time, int(args.num_time_points))
    trajs = generate_noise_trajs(args.noise_t_max, args.noise_dt, args.num_realizations, args.S0)
    
    for case_name, ops, A in [("undriven", ops_undriven, 0.0), ("driven", ops_driven, driven_A)]:
        print(f"\n=== Simulating {case_name.upper()} Case ===")
        
        c_ops = [
            np.sqrt(args.gamma_main) * ops["sop_total"],
            np.sqrt(args.gamma_main) * ops["c1op_total"],
            np.sqrt(args.gamma_main) * ops["c2op_total"],
            np.sqrt(args.gamma_aux) * ops["a_c3_total"],
            np.sqrt(args.gamma_aux) * ops["a_c4_total"],
        ]
        
        n_traj = len(trajs)
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(simulate_single_trajectory)(
                i, trajs, args.noise_dt, 100000,
                ops["sds_total"], ops["H0_total"], psi0, time_points, c_ops,
            )
            for i in range(n_traj)
        )

        rho_sum = None
        for rho_arr in results:
            if rho_sum is None:
                rho_sum = rho_arr.copy()
            else:
                rho_sum += rho_arr
        avg_rho_t = rho_sum / n_traj
        
        rho_final_full = qt.Qobj(avg_rho_t[-1], dims=ops["H0_total"].dims)
        
        dressed = build_dressed_subspace_from_h0(ops["H0_total"], trunc_dim)
        perform_tomography_and_plot(
            rho_final_full, 
            dressed["dressed_kets"], 
            f"Final State Tomography ({case_name}, t={args.sim_time/1000:.0f}us)", 
            f"tomo_final_{case_name}.png"
        )

if __name__ == "__main__":
    main()
