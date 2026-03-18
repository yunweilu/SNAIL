import argparse
import os
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from qutip import matrix_histogram
from joblib import Parallel, delayed

import sys
sys.path.insert(0, "/home/yunwei/SNAIL/full_simulation copy/dual_rail/single_dual_rial")
from noise_generator import GenerateNoise

sys.path.insert(0, "/home/yunwei/SNAIL/full_simulation copy/dual_rail/two_dual_rails")
from importlib.machinery import SourceFileLoader
twodualrail = SourceFileLoader("2dualrail", "/home/yunwei/SNAIL/full_simulation copy/dual_rail/two_dual_rails/2dualrail.py").load_module()


def basis_index(t, c2, c1, c3, c4, trunc_dim):
    n_t, n_c2, n_c1 = trunc_dim
    n_c3 = 2
    n_c4 = 2
    return t * (n_c2 * n_c1 * n_c3 * n_c4) + c2 * (n_c1 * n_c3 * n_c4) + c1 * (n_c3 * n_c4) + c3 * n_c4 + c4

def extract_dualrail_A(Op_total, trunc_dim):
    """
    Extracts subsystem A (t, c1, c3) by projecting the 5-mode operator onto the subspace
    where c2=0 and c4=0.
    """
    n_t, n_c2, n_c1 = trunc_dim
    n_c3 = 2
    n_c4 = 2
    
    dim_A = n_t * n_c1 * n_c3
    Op_A_data = np.zeros((dim_A, dim_A), dtype=complex)
    Op_full = Op_total.full()
    
    for tA in range(n_t):
        for c1A in range(n_c1):
            for c3A in range(n_c3):
                row_full = basis_index(t=tA, c2=0, c1=c1A, c3=c3A, c4=0, trunc_dim=trunc_dim)
                row_A = tA * (n_c1 * n_c3) + c1A * n_c3 + c3A
                
                for tB in range(n_t):
                    for c1B in range(n_c1):
                        for c3B in range(n_c3):
                            col_full = basis_index(t=tB, c2=0, c1=c1B, c3=c3B, c4=0, trunc_dim=trunc_dim)
                            col_A = tB * (n_c1 * n_c3) + c1B * n_c3 + c3B
                            
                            Op_A_data[row_A, col_A] = Op_full[row_full, col_full]
                            
    return qt.Qobj(Op_A_data, dims=[[n_t, n_c1, n_c3], [n_t, n_c1, n_c3]])

def build_dressed_subspace_from_h0_A(H0_A, trunc_dim):
    H_arr = np.asarray(H0_A.full(), dtype=complex)
    evals, U = np.linalg.eigh(H_arr)
    
    n_t, n_c2, n_c1 = trunc_dim
    n_c3 = 2
    
    # Basis order is t, c1, c3
    label_to_index = {
        "0_L": 0 * (n_c1 * n_c3) + 1 * n_c3 + 0, # t=0, c1=1, c3=0
        "1_L": 0 * (n_c1 * n_c3) + 0 * n_c3 + 1, # t=0, c1=0, c3=1
    }

    dims_ket = [[n_t, n_c1, n_c3], [1, 1, 1]]
    dressed_kets = {}
    dressed_projectors = {}
    matched_cols = {}
    used_cols = set()
    for lbl, bare_idx in label_to_index.items():
        if bare_idx >= U.shape[0]:
            continue
        overlaps = np.abs(U[bare_idx, :]) ** 2
        order = np.argsort(overlaps)[::-1]
        col = None
        for cand in order:
            if int(cand) not in used_cols:
                col = int(cand)
                break
        if col is None:
            continue
        used_cols.add(col)
        matched_cols[lbl] = col
        ket = qt.Qobj(U[:, col], dims=dims_ket)
        dressed_kets[lbl] = ket
        dressed_projectors[lbl] = ket * ket.dag()

    return {
        "evals": evals,
        "U": U,
        "matched_cols": matched_cols,
        "dressed_kets": dressed_kets,
        "dressed_projectors": dressed_projectors,
        "labels": ["0_L", "1_L"],
    }

def build_initial_state_A(trunc_dim):
    n_t, n_c2, n_c1 = trunc_dim
    n_c3 = 2
    
    # + state for logical qubit A
    # 0_L = |0, 1, 0> in (t, c1, c3)
    ket0 = qt.tensor(qt.basis(n_t, 0), qt.basis(n_c1, 1), qt.basis(n_c3, 0))
    # 1_L = |0, 0, 1> in (t, c1, c3)
    ket1 = qt.tensor(qt.basis(n_t, 0), qt.basis(n_c1, 0), qt.basis(n_c3, 1))
    
    return (ket0 + ket1).unit()


def build_dressed_initial_state_A(trunc_dim, dressed_kets):
    if "0_L" in dressed_kets and "1_L" in dressed_kets:
        return (dressed_kets["0_L"] + dressed_kets["1_L"]).unit()
    return build_initial_state_A(trunc_dim)

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
        "nsteps": int(5*solver_nsteps),
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

def logical_density_from_full_rho(rho_full, dressed_kets):
    rho_logical_arr = np.zeros((2, 2), dtype=complex)
    labels = ["0_L", "1_L"]
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if li in dressed_kets and lj in dressed_kets:
                val = dressed_kets[li].dag() * (rho_full * dressed_kets[lj])
                if isinstance(val, qt.Qobj):
                    rho_logical_arr[i, j] = val.full()[0, 0]
                else:
                    rho_logical_arr[i, j] = complex(val)
    rho_logical = qt.Qobj(rho_logical_arr, dims=[[2], [2]])
    if np.abs(rho_logical.tr()) > 0:
        rho_logical = rho_logical / rho_logical.tr()
    return rho_logical

def perform_case_comparison_triptych(case_data, initial_logical_rho, sim_time_ns, filename):
    """Plot initial, undriven final, and driven final in one figure."""
    labels = ["|01>", "|10>"]
    rho_undriven = case_data["undriven"]["logical_rhos"][-1]
    rho_driven = case_data["driven"]["logical_rhos"][-1]
    t_us = sim_time_ns / 1000.0
    panels = [
        ("Initial", initial_logical_rho),
        (f"Undriven Final (t={t_us:.0f} us)", rho_undriven),
        (f"Driven Final (t={t_us:.0f} us)", rho_driven),
    ]
    fig = plt.figure(figsize=(13.2, 4.6))
    axes = []

    for panel_idx, (panel_title, panel_rho) in enumerate(panels):
        rho_abs = qt.Qobj(np.abs(panel_rho.full()), dims=panel_rho.dims)
        ax = fig.add_subplot(1, 3, panel_idx + 1, projection="3d")
        matrix_histogram(
            rho_abs,
            labels,
            labels,
            fig=fig,
            ax=ax,
            limits=[0.0, 0.5],
            bar_style="abs",
            color_style="abs",
            color_limits=[0.0, 0.5],
            cmap=mpl.cm.jet,
            colorbar=False,
        )
        ax.set_title(panel_title)
        ax.view_init(azim=-55, elev=45)
        ax.set_zlim(0.0, 0.5)
        axes.append(ax)

    shared_norm = mpl.colors.Normalize(vmin=0.0, vmax=0.5)
    shared_mappable = mpl.cm.ScalarMappable(norm=shared_norm, cmap=mpl.cm.jet)
    shared_mappable.set_array([])
    fig.subplots_adjust(wspace=0.2, right=0.87)
    cax = fig.add_axes([0.89, 0.2, 0.015, 0.62])
    cbar = fig.colorbar(shared_mappable, cax=cax)
    cbar.set_label("Matrix element magnitude")

    plot_out = Path(filename).resolve()
    fig.savefig(plot_out, dpi=200, bbox_inches="tight")
    print(f"Saved logical tomography triptych to: {plot_out}")

def main():
    parser = argparse.ArgumentParser(description="Dual-rail A dynamics")
    parser.add_argument(
        "--A-over-pi",
        type=float,
        default=10e-3,
        help="Drive amplitude factor with A = (A-over-pi) * 2*pi (matches static_sweep).",
    )
    parser.add_argument("--sim-time", type=float, default=60000.0, help="Simulation end time.")
    parser.add_argument("--num-time-points", type=int, default=10, help="Number of saved density-matrix points.")
    parser.add_argument("--num-realizations", type=int, default=20, help="Number of noise trajectories.")
    parser.add_argument("--S0", type=float, default=1e-5, help="Noise amplitude.")
    parser.add_argument("--noise-t-max", type=float, default=int(1e8), help="Tracking max time")
    parser.add_argument("--noise-dt", type=float, default=1000.0, help="Tracking dt")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel processing.")
    parser.add_argument("--gamma-main", type=float, default=1 / (2e4), help="Decay main")
    parser.add_argument("--gamma-aux", type=float, default=5e-7, help="Decay aux")
    parser.add_argument("--phi-ex", type=float, default=0.2, help="External flux bias used for dc/dPhi.")
    parser.add_argument("--detuning-mhz", type=float, default=28.5, help="Drive detuning (MHz) for dc/dPhi.")
    args = parser.parse_args()

    # We build the 5-mode Hamiltonian first
    ops_undriven_full = twodualrail.build_total_operators(A=0.0)
    driven_A = args.A_over_pi * 2 * np.pi
    ops_driven_full = twodualrail.build_total_operators(A=driven_A)
    
    trunc_dim = ops_undriven_full.get("trunc_dim", [3, 2, 2])
    

    time_points = np.linspace(0.0, args.sim_time, int(args.num_time_points))
    trajs = generate_noise_trajs(args.noise_t_max, args.noise_dt, args.num_realizations, args.S0)
    
    case_data = {}
    for case_name, ops_full, A in [("undriven", ops_undriven_full, 0.0), ("driven", ops_driven_full, driven_A)]:
        print(f"\n=== Simulating Dual-Rail A {case_name.upper()} Case ===")
        dc1_dphi, dc2_dphi = twodualrail.compute_dc_dphi(
            phi_ex=args.phi_ex, detuning_mhz=args.detuning_mhz, A=A
        )
        print(
            f"dc1/dPhi: {dc1_dphi:.6f} GHz/Phi0 | dc2/dPhi: {dc2_dphi:.6f} GHz/Phi0 "
            f"(detuning={args.detuning_mhz:.1f} MHz, phi_ex={args.phi_ex:.3f})"
        )
        print(
            f"Dual-rail A mapping: rail-1(c1)={dc1_dphi:.6f} GHz/Phi0, "
            f"rail-2(c2)={dc2_dphi:.6f} GHz/Phi0"
        )
        
        # Truncate operators to Dual-Rail A
        ops_A = {
            "H0_total": extract_dualrail_A(ops_full["H0_total"], trunc_dim),
            "sds_total": extract_dualrail_A(ops_full["sds_total"], trunc_dim),
            "sop_total": extract_dualrail_A(ops_full["sop_total"], trunc_dim),
            "c1op_total": extract_dualrail_A(ops_full["c1op_total"], trunc_dim),
            "a_c3_total": extract_dualrail_A(ops_full["a_c3_total"], trunc_dim),
        }

        dressed_A = build_dressed_subspace_from_h0_A(ops_A["H0_total"], trunc_dim)
        psi0_A = build_dressed_initial_state_A(trunc_dim, dressed_A["dressed_kets"])
        
        c_ops_A = [
            np.sqrt(args.gamma_main) * ops_A["sop_total"],
            np.sqrt(args.gamma_main) * ops_A["c1op_total"],
            np.sqrt(args.gamma_aux) * ops_A["a_c3_total"]
            # Note: We omit c2op and a_c4 because they act on dual-rail B
        ]
        
        n_traj = len(trajs)
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(simulate_single_trajectory)(
                i, trajs, args.noise_dt, 100000,
                ops_A["sds_total"], ops_A["H0_total"], psi0_A, time_points, c_ops_A,
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
        
        logical_rhos = []
        
        print("Collecting logical density matrices over time...")
        for rho_arr in avg_rho_t:
            rho_full = qt.Qobj(rho_arr, dims=ops_A["H0_total"].dims)
            rho_logic = logical_density_from_full_rho(rho_full, dressed_A["dressed_kets"])
            logical_rhos.append(rho_logic)

        case_data[case_name] = {
            "logical_rhos": logical_rhos,
            "final_full_rho": qt.Qobj(avg_rho_t[-1], dims=ops_A["H0_total"].dims)
        }
        
        with open(f"final_data_A_{case_name}.pkl", "wb") as f:
            pickle.dump({
                "time_points": time_points,
                "logical_rhos": logical_rhos,
                "avg_rho_t_last": avg_rho_t[-1]
            }, f)
            print(f"Saved final data to final_data_A_{case_name}.pkl")

    initial_logical_rho = case_data["driven"]["logical_rhos"][0]
    perform_case_comparison_triptych(
        case_data,
        initial_logical_rho,
        args.sim_time,
        "tomo_final_A_comparison.png",
    )

if __name__ == "__main__":
    main()
