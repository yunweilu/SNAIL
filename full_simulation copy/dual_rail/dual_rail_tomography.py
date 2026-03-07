import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from qutip import matrix_histogram

from system import sort_eigenpairs


def basis_index(t, c, a, n_cavity=2, n_aux=2):
    return t * (n_cavity * n_aux) + c * n_aux + a


def build_dressed_subspace_from_h0(H0_total, n_transmon=3, n_cavity=2, n_aux=2):
    H_arr = np.asarray(H0_total.full(), dtype=complex)
    evals, U = np.linalg.eigh(H_arr)
    evals, U = sort_eigenpairs(evals, U)

    label_list = ["001", "010", "101", "110", "201", "210"]
    label_to_index = {
        "001": basis_index(0, 0, 1, n_cavity=n_cavity, n_aux=n_aux),
        "010": basis_index(0, 1, 0, n_cavity=n_cavity, n_aux=n_aux),
        "101": basis_index(1, 0, 1, n_cavity=n_cavity, n_aux=n_aux),
        "110": basis_index(1, 1, 0, n_cavity=n_cavity, n_aux=n_aux),
        "201": basis_index(2, 0, 1, n_cavity=n_cavity, n_aux=n_aux),
        "210": basis_index(2, 1, 0, n_cavity=n_cavity, n_aux=n_aux),
    }

    dims_ket = [[n_transmon, n_cavity, n_aux], [1, 1, 1]]
    dressed_kets = {}
    dressed_projectors = {}
    for lbl in label_list:
        idx = label_to_index[lbl]
        ket = qt.Qobj(U[:, idx], dims=dims_ket)
        dressed_kets[lbl] = ket
        dressed_projectors[lbl] = ket * ket.dag()

    dual_rail_projector = 0 * dressed_projectors[label_list[0]]
    for lbl in label_list:
        dual_rail_projector = dual_rail_projector + dressed_projectors[lbl]

    return {
        "evals": evals,
        "U": U,
        "dressed_kets": dressed_kets,
        "dressed_projectors": dressed_projectors,
        "dual_rail_projector": dual_rail_projector,
        "labels": label_list,
    }


def logical_density_from_full_rho(rho_full, logical0_ket, logical1_ket):
    rho00 = complex(logical0_ket.overlap(rho_full * logical0_ket))
    rho01 = complex(logical0_ket.overlap(rho_full * logical1_ket))
    rho10 = complex(logical1_ket.overlap(rho_full * logical0_ket))
    rho11 = complex(logical1_ket.overlap(rho_full * logical1_ket))
    rho_logical = qt.Qobj(
        np.array([[rho00, rho01], [rho10, rho11]], dtype=complex),
        dims=[[2], [2]],
    )
    return rho_logical


def logical_density_from_reduced_ca(rho_ca):
    """rho_ca is 4x4 on (cavity, aux) with logical basis |01>,|10>."""
    arr = np.asarray(rho_ca.full(), dtype=complex)
    rho_logical_arr = np.zeros((2, 2), dtype=complex)
    rho_logical_arr[0, 0] = arr[1, 1]  # <01|rho|01>
    rho_logical_arr[0, 1] = arr[1, 2]  # <01|rho|10>
    rho_logical_arr[1, 0] = arr[2, 1]  # <10|rho|01>
    rho_logical_arr[1, 1] = arr[2, 2]  # <10|rho|10>
    return qt.Qobj(rho_logical_arr, dims=[[2], [2]])


def plot_logical_tomography_over_time(time_points, rho_logical_t, case_name, plot_prefix, dpi=200, n_plot=10):
    n_t = len(time_points)
    n_plot = min(max(2, int(n_plot)), n_t)
    if n_t <= n_plot:
        idx = np.arange(n_t, dtype=int)
    else:
        idx = np.linspace(0, n_t - 1, n_plot, dtype=int)
    t_plot = np.asarray(time_points, dtype=float)[idx]
    rho_plot = np.asarray(rho_logical_t, dtype=complex)[idx]

    for k, ti in enumerate(t_plot):
        rho_k = qt.Qobj(rho_plot[k], dims=[[2], [2]])
        fig, ax = matrix_histogram(rho_k, ["|01>", "|10>"], ["|01>", "|10>"])
        ax.set_title(f"Logical Subspace ({case_name}), t={ti:.2f} ns")
        ax.view_init(azim=-55, elev=45)
        plt.tight_layout()
        out = Path(f"{plot_prefix}_{case_name}_{n_plot}points_t{k:02d}.png")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved sampled logical histogram to: {out.resolve()}")
        plt.show()


def run_tomography_from_arrays(
    case_name,
    A,
    time_points,
    avg_rho_t,
    H0_total,
    plot_prefix,
    dpi=200,
    data_label="in-memory",
    n_plot_points=10,
):
    n_transmon, n_cavity, n_aux = 3, 2, 2

    print(f"\n=== Case: {case_name} ===")
    print(f"A = {A:.6e}")
    print("Loaded data source:", data_label)
    print("Hamiltonian used for dressed-state labeling (H0_total):")
    print(H0_total)
    print("Projecting density matrices after tracing out transmon subsystem.")

    dressed = build_dressed_subspace_from_h0(
        H0_total, n_transmon=n_transmon, n_cavity=n_cavity, n_aux=n_aux
    )
    P_dual = dressed["dual_rail_projector"]
    rho_logical_list = []
    for k in range(len(time_points)):
        rho_full = qt.Qobj(
            avg_rho_t[k],
            dims=[[n_transmon, n_cavity, n_aux], [n_transmon, n_cavity, n_aux]],
        )
        _rho_dual = P_dual * rho_full * P_dual
        rho_ca = rho_full.ptrace([1, 2])  # trace out transmon (subsystem 0)
        rho_logical_k = logical_density_from_reduced_ca(rho_ca)
        rho_logical_list.append(rho_logical_k.full())
    rho_logical_t = np.asarray(rho_logical_list)
    rho_logical = qt.Qobj(rho_logical_t[-1], dims=[[2], [2]])

    print("\nDressed projectors included in dual-rail subspace:")
    for lbl in dressed["labels"]:
        print(f"  |{lbl}><{lbl}|")
    print("Logical mapping: |0_L> := |001>, |1_L> := |010> (dressed labels)")
    print("\nFinal logical 2x2 density matrix:")
    print(rho_logical)

    fig, ax = matrix_histogram(rho_logical, ["|01>", "|10>"], ["|01>", "|10>"])
    ax.set_title(f"Logical Subspace ({case_name}) at t={float(time_points[-1]):.0f} ns")
    ax.view_init(azim=-55, elev=45)
    plt.tight_layout()
    plot_out = Path(plot_prefix + f"_{case_name}.png")
    fig.savefig(plot_out, dpi=dpi, bbox_inches="tight")
    print(f"Saved logical tomography plot to: {plot_out.resolve()}")
    plt.show()
    plot_logical_tomography_over_time(
        np.asarray(time_points),
        rho_logical_t,
        case_name=case_name,
        plot_prefix=plot_prefix,
        dpi=dpi,
        n_plot=n_plot_points,
    )

    return {
        "A": A,
        "time_points": np.asarray(time_points),
        "avg_rho_t": avg_rho_t,
        "rho_logical_t": rho_logical_t,
        "rho_logical_final": rho_logical.full(),
    }


def run_case_from_loaded_data(case_name, A, time_points, avg_rho_t, args):
    from dual_rail import build_total_operators

    H0_total = build_total_operators(
        A=A,
        n_transmon=3,
        n_cavity=2,
        n_aux=2,
    )["H0_total"]
    return run_tomography_from_arrays(
        case_name=case_name,
        A=A,
        time_points=time_points,
        avg_rho_t=avg_rho_t,
        H0_total=H0_total,
        plot_prefix=args.plot_prefix,
        dpi=args.dpi,
        data_label=str(Path(args.data).resolve()),
        n_plot_points=10,
    )


def main():
    parser = argparse.ArgumentParser(description="Dual-rail dressed-subspace tomography from saved data.")
    parser.add_argument(
        "--data",
        type=str,
        default="dual_rail_density_over_time.pkl",
        help="Input density-matrix data from dual_rail.py",
    )
    parser.add_argument("--case", choices=["undriven", "driven", "both", "auto"], default="auto")
    parser.add_argument("--A-over-pi-fallback", type=float, default=0.0, help="Fallback A/pi when not present in data.")
    parser.add_argument("--output", type=str, default="dual_rail_tomography.pkl", help="Output pickle path.")
    parser.add_argument("--plot-prefix", type=str, default="dual_rail_tomography", help="Output figure prefix.")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    args = parser.parse_args()

    with Path(args.data).open("rb") as f:
        src = pickle.load(f)

    run_cases = []
    if "time_points" in src and "avg_rho_t" in src:
        # Single-case format from dual_rail.py
        params = src.get("params", {})
        A_val = params.get("A", None)
        if A_val is None:
            A_over_pi = params.get("A_over_pi", params.get("A-over-pi", args.A_over_pi_fallback))
            A_val = float(A_over_pi) * np.pi
        run_cases.append(("loaded", float(A_val), np.asarray(src["time_points"]), np.asarray(src["avg_rho_t"])))
    else:
        # Multi-case format
        available = [k for k in ("undriven", "driven") if k in src]
        if not available:
            raise ValueError("Input data file has no recognized density-matrix payload.")
        if args.case == "auto":
            selected = available
        elif args.case == "both":
            selected = available
        else:
            if args.case not in src:
                raise ValueError(f"Requested case '{args.case}' not found. Available: {available}")
            selected = [args.case]
        for c in selected:
            A_val = float(src[c].get("A", 0.0))
            run_cases.append((c, A_val, np.asarray(src[c]["time_points"]), np.asarray(src[c]["avg_rho_t"])))

    payload = {"params": vars(args)}
    for case_name, A, t_arr, rho_arr in run_cases:
        payload[case_name] = run_case_from_loaded_data(case_name, A, t_arr, rho_arr, args)

    out_path = Path(args.output)
    with out_path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved tomography payload to: {out_path.resolve()}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
