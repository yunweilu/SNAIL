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
    
    # 00_LL = 0, 1, 1, 0, 0
    ket00 = qt.tensor(qt.basis(n_t, 0), qt.basis(n_c2, 1), qt.basis(n_c1, 1), qt.basis(n_c3, 0), qt.basis(n_c4, 0))
    # 11_LL = 0, 0, 0, 1, 1
    ket11 = qt.tensor(qt.basis(n_t, 0), qt.basis(n_c2, 0), qt.basis(n_c1, 0), qt.basis(n_c3, 1), qt.basis(n_c4, 1))
    
    return (ket00 + ket11).unit()

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
    # Absolute value of the density matrix
    rho_abs = qt.Qobj(np.abs(rho_logical.full()))
    fig, ax = matrix_histogram(rho_abs, labels, labels)
    
    ax.set_title(title)
    ax.view_init(azim=-55, elev=45)
    plt.tight_layout()
    plot_out = Path(filename).resolve()
    fig.savefig(plot_out, dpi=200, bbox_inches="tight")
    print(f"Saved logical tomography plot to: {plot_out}")


def perform_case_comparison_triptych(case_data, initial_logical_rho, sim_time_ns, filename):
    """Plot initial, undriven final, and driven final in one figure."""
    labels = ["|00>", "|01>", "|10>", "|11>"]
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
        ax = fig.add_subplot(1, 3, panel_idx + 1, projection="3d")
        matrix_histogram(
            panel_rho,
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
    parser = argparse.ArgumentParser(description="Two Dual-rails dynamics")
    parser.add_argument(
        "--A-over-pi",
        type=float,
        default=10e-3,
        help="Drive amplitude factor with A = (A-over-pi) * 2*pi (matches dual_raila).",
    )
    parser.add_argument("--sim-time", type=float, default=1.0, help="Simulation end time.")
    parser.add_argument("--num-time-points", type=int, default=10, help="Number of saved density-matrix points.")
    parser.add_argument("--num-realizations", type=int, default=20, help="Number of noise trajectories.")
    parser.add_argument("--S0", type=float, default=1e-5, help="Noise amplitude.")
    parser.add_argument("--noise-t-max", type=float, default=int(1e8), help="Tracking max time")
    parser.add_argument("--noise-dt", type=float, default=1000.0, help="Tracking dt")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel processing workers. -1 uses auto mode with a safe cap.",
    )
    parser.add_argument("--gamma-main", type=float, default=1 / (2e4), help="Decay main")
    parser.add_argument("--gamma-aux", type=float, default=5e-7, help="Decay aux")
    parser.add_argument("--phi-ex", type=float, default=0.2, help="External flux bias used for dc/dPhi.")
    parser.add_argument("--detuning-mhz", type=float, default=28.5, help="Drive detuning (MHz) for dc/dPhi.")
    args = parser.parse_args()

    driven_A = args.A_over_pi * 2 * np.pi
    dc1_dphi, dc2_dphi = twodualrail.compute_dc_dphi(
        phi_ex=args.phi_ex, detuning_mhz=args.detuning_mhz, A=driven_A
    )
    print(
        f"dc1/dPhi: {dc1_dphi:.6f} GHz/Phi0 | dc2/dPhi: {dc2_dphi:.6f} GHz/Phi0 "
        f"(detuning={args.detuning_mhz:.1f} MHz, phi_ex={args.phi_ex:.3f})"
    )

    ops_undriven = twodualrail.build_total_operators(A=0.0)
    ops_driven = twodualrail.build_total_operators(A=driven_A)
    
    trunc_dim = ops_undriven.get("trunc_dim", [3, 2, 2])
    
    psi0 = build_initial_state(trunc_dim)
    dressed_undriven = build_dressed_subspace_from_h0(ops_undriven["H0_total"], trunc_dim)
    
    time_points = np.linspace(0.0, args.sim_time, int(args.num_time_points))
    trajs = generate_noise_trajs(args.noise_t_max, args.noise_dt, args.num_realizations, args.S0)
    
    case_data = {}
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
        
        dressed = build_dressed_subspace_from_h0(ops["H0_total"], trunc_dim)
        
        logical_rhos = []
        
        print("Collecting logical density matrices over time...")
        for rho_arr in avg_rho_t:
            rho_full = qt.Qobj(rho_arr, dims=ops["H0_total"].dims)
            rho_logic = logical_density_from_full_rho(rho_full, dressed["dressed_kets"])
            logical_rhos.append(rho_logic)

        case_data[case_name] = {
            "logical_rhos": logical_rhos,
            "final_full_rho": qt.Qobj(avg_rho_t[-1], dims=ops["H0_total"].dims)
        }
        print(f"Final logical density matrix ({case_name}):")
        print(logical_rhos[-1].full())

        with open(f"final_data_{case_name}.pkl", "wb") as f:
            pickle.dump({
                "time_points": time_points,
                "logical_rhos": logical_rhos,
                "avg_rho_t_last": avg_rho_t[-1]
            }, f)
            print(f"Saved final data to final_data_{case_name}.pkl")

    # Initial logical Bell state |Phi+> = (|00> + |11>) / sqrt(2)
    psi_ideal_log = (qt.basis(4, 0) + qt.basis(4, 3)).unit()
    rho_ideal_log = psi_ideal_log * psi_ideal_log.dag()
    perform_case_comparison_triptych(
        case_data,
        rho_ideal_log,
        args.sim_time,
        "tomo_final_comparison.png",
    )

if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
