import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from joblib import Parallel, delayed

from noise_generator import GenerateNoise
from hamiltonian_generator import Hamiltonian
from dual_rail_tomography import run_tomography_from_arrays
from system import sort_eigenpairs


def annihilation(dim):
    return np.diag(np.sqrt(np.arange(1, dim)), 1)


def build_tc_operators(A, n_transmon=3, n_cavity=2):
    phi_ex = 0.2
    Ej = 30.19
    Ec = 0.1
    tol = 1e-12

    sc_tmp = Hamiltonian(phi_ex, Ej, Ec, [5, 10])
    optimal_omega, _ = sc_tmp.optimal_omegad(A)
    optimal_omega = optimal_omega * 2 * np.pi

    sc = Hamiltonian(phi_ex, Ej, Ec, [n_transmon, n_cavity])
    total_dim = n_transmon * n_cavity

    a_s = np.kron(annihilation(n_transmon), np.eye(n_cavity))
    a_c = np.kron(np.eye(n_transmon), annihilation(n_cavity))
    n_s = np.kron(np.diag(np.arange(n_transmon)), np.eye(n_cavity))
    n_c = np.kron(np.eye(n_transmon), np.diag(np.arange(n_cavity)))

    sds_raw = np.asarray(sc.noise, dtype=complex).copy()
    diag_vals = np.diag(sds_raw) - sds_raw[0, 0]
    np.fill_diagonal(sds_raw, diag_vals)
    sds_diag = np.zeros_like(sds_raw, dtype=complex)
    diag_mask = np.eye(sds_raw.shape[0], dtype=bool)
    sds_diag[diag_mask] = sds_raw[diag_mask]

    sop_raw = np.asarray(sc.s, dtype=complex)
    s_mask = np.abs(a_s) > tol
    c_mask = np.abs(a_c) > tol
    sop_selected = np.zeros_like(sop_raw, dtype=complex)
    cop_selected = np.zeros_like(sop_raw, dtype=complex)
    sop_selected[s_mask] = sop_raw[s_mask]
    cop_selected[c_mask] = sop_raw[c_mask]

    H_control_filtered = np.zeros_like(np.asarray(sc.H_control), dtype=complex)
    for i in range(total_dim - n_cavity):
        j = i + n_cavity
        H_control_filtered[i, j] = sc.H_control[i, j]
        H_control_filtered[j, i] = sc.H_control[j, i]

    H0_diag = np.diag(np.diag(sc.H) - sc.H[0, 0])
    omegac = np.abs(H0_diag[1, 1]) if H0_diag.shape[0] > 1 else 0.0
    H0_rot_raw = H0_diag - optimal_omega * n_s - omegac * n_c + (A / 2.0) * H_control_filtered
    evals_rot, U = np.linalg.eigh(H0_rot_raw)
    evals_rot, U = sort_eigenpairs(evals_rot, U)
    evals_rot = np.real(evals_rot - evals_rot[0])
    omega_c = evals_rot[1] if len(evals_rot) > 1 else 0.0
    H0_rot_raw = H0_rot_raw - omega_c * n_c

    return {
        "phi_ex": phi_ex,
        "optimal_omega": optimal_omega,
        "sds_diag": sds_diag,
        "sop_selected": sop_selected,
        "cop_selected": cop_selected,
        "H0_rot_raw": H0_rot_raw,
    }


def build_total_operators(A, n_transmon=3, n_cavity=2, n_aux=2):
    tc = build_tc_operators(A=A, n_transmon=n_transmon, n_cavity=n_cavity)
    dim_tc = n_transmon * n_cavity
    I_aux = np.eye(n_aux, dtype=complex)
    I_tc = np.eye(dim_tc, dtype=complex)

    H0_total = np.kron(tc["H0_rot_raw"], I_aux)
    sds_total = np.kron(tc["sds_diag"], I_aux)
    sop_total = np.kron(tc["sop_selected"], I_aux)
    cop_total = np.kron(tc["cop_selected"], I_aux)
    a_aux = np.kron(I_tc, annihilation(n_aux))

    dims = [[n_transmon, n_cavity, n_aux], [n_transmon, n_cavity, n_aux]]
    return {
        "phi_ex": tc["phi_ex"],
        "optimal_omega": tc["optimal_omega"],
        "H0_total": qt.Qobj(H0_total, dims=dims),
        "sds_total": qt.Qobj(sds_total, dims=dims),
        "sop_total": qt.Qobj(sop_total, dims=dims),
        "cop_total": qt.Qobj(cop_total, dims=dims),
        "a_aux_total": qt.Qobj(a_aux, dims=dims),
    }


def build_initial_plus_state(n_transmon=3, n_cavity=2, n_aux=2):
    ket001 = qt.tensor(qt.basis(n_transmon, 0), qt.basis(n_cavity, 0), qt.basis(n_aux, 1))
    ket010 = qt.tensor(qt.basis(n_transmon, 0), qt.basis(n_cavity, 1), qt.basis(n_aux, 0))
    return (ket001 + ket010).unit()


def simulate_single_trajectory(
    traj_idx,
    raw_trajs,
    noise_dt,
    solver_nsteps,
    sds_total,
    H0_total,
    psi0,
    time_points,
    c_ops,
):
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


def generate_noise_trajs(noise_t_max, noise_dt, num_realizations, S0):
    n_noise_samples = int(np.ceil(float(noise_t_max) / float(noise_dt)))
    if n_noise_samples <= 0:
        raise ValueError("Computed noise sample count must be positive.")
    gn = GenerateNoise(
        1,
        n_noise_samples,
        S0**2,
        num_realizations,
        False,
    )
    return gn.generate_colored_noise()


def basis_index(t, c, a, n_cavity, n_aux):
    return t * (n_cavity * n_aux) + c * n_aux + a


def plot_dual_rail_populations(time_points, avg_rho_t, n_cavity=2, n_aux=2, save_path=None, dpi=200):
    idx_001 = basis_index(0, 0, 1, n_cavity, n_aux)
    idx_010 = basis_index(0, 1, 0, n_cavity, n_aux)
    p001 = np.real(avg_rho_t[:, idx_001, idx_001])
    p010 = np.real(avg_rho_t[:, idx_010, idx_010])

    plt.figure(figsize=(7, 4))
    plt.plot(time_points, p001, label="P(|001>)")
    plt.plot(time_points, p010, label="P(|010>)")
    plt.plot(time_points, p001 + p010, "--", label="P(logical subspace)")
    plt.xlabel("t (ns)")
    plt.ylabel("Population")
    plt.title("Dual-rail populations vs time")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved population plot to: {out.resolve()}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Dual-rail dynamics with tc(3x2) plus ancilla(2).")
    parser.add_argument("--case", choices=["undriven", "driven", "both"], default="both", help="Which case(s) to simulate.")
    parser.add_argument("--A-over-pi", type=float, default=10e-3, help="Drive amplitude as A/pi for driven case.")
    parser.add_argument("--sim-time", type=float, default=100000.0, help="Simulation end time.")
    parser.add_argument("--num-time-points", type=int, default=100, help="Number of saved density-matrix points.")
    parser.add_argument("--num-realizations", type=int, default=100, help="Number of noise trajectories.")
    parser.add_argument("--S0", type=float, default=1e-5, help="Noise amplitude.")
    parser.add_argument("--noise-t-max", type=float, default=int(1e8), help="Trajectory-generation t_max.")
    parser.add_argument("--noise-dt", type=float, default=1000.0, help="Trajectory dt.")
    parser.add_argument("--solver-nsteps", type=int, default=100000, help="Mesolve max nsteps.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs.")
    parser.add_argument("--gamma-main", type=float, default=1 / (2e4), help="Rate for s/c collapse operators.")
    parser.add_argument("--gamma-aux", type=float, default=5e-7, help="Ancilla decay rate.")
    parser.add_argument(
        "--output",
        type=str,
        default="dual_rail_density_over_time.pkl",
        help="Output pickle for averaged density matrices.",
    )
    parser.add_argument(
        "--plot-save-path",
        type=str,
        default="dual_rail_populations.png",
        help="Output path for population plot.",
    )
    args = parser.parse_args()

    if args.noise_dt <= 0:
        raise ValueError("noise_dt must be > 0")
    if args.num_time_points < 2:
        raise ValueError("num_time_points must be >= 2")

    n_transmon, n_cavity, n_aux = 3, 2, 2
    time_points = np.linspace(0.0, args.sim_time, int(args.num_time_points))
    trajs = generate_noise_trajs(args.noise_t_max, args.noise_dt, args.num_realizations, args.S0)
    psi0 = build_initial_plus_state(n_transmon=n_transmon, n_cavity=n_cavity, n_aux=n_aux)

    run_cases = []
    if args.case in ("both", "undriven"):
        run_cases.append(("undriven", 0.0))
    if args.case in ("both", "driven"):
        run_cases.append(("driven", args.A_over_pi * np.pi))

    payload = {
        "params": vars(args),
        "dims": [n_transmon, n_cavity, n_aux],
    }

    for case_name, A in run_cases:
        ops = build_total_operators(A=A, n_transmon=n_transmon, n_cavity=n_cavity, n_aux=n_aux)
        c_ops = [
            np.sqrt(args.gamma_main) * ops["sop_total"],
            np.sqrt(args.gamma_main) * ops["cop_total"],
            np.sqrt(args.gamma_aux) * ops["a_aux_total"],
        ]

        print(f"\n=== Case: {case_name} ===")
        print(f"A = {A:.6e}")
        print("Initial state psi0:")
        print(psi0)
        print("\nHamiltonian H0_total:")
        print(ops["H0_total"])
        print("\nNoise operator sds_total:")
        print(ops["sds_total"])
        print("\nCollapse operators:")
        for i, c in enumerate(c_ops):
            print(f"c_ops[{i}] =")
            print(c)

        n_traj = len(trajs)
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(simulate_single_trajectory)(
                i,
                trajs,
                args.noise_dt,
                args.solver_nsteps,
                ops["sds_total"],
                ops["H0_total"],
                psi0,
                time_points,
                c_ops,
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

        payload[case_name] = {
            "A": A,
            "time_points": time_points,
            "avg_rho_t": avg_rho_t,
        }

        pop_out = Path(args.plot_save_path)
        pop_case_out = str(pop_out.with_name(f"{pop_out.stem}_{case_name}{pop_out.suffix}"))
        plot_dual_rail_populations(
            time_points=time_points,
            avg_rho_t=avg_rho_t,
            n_cavity=n_cavity,
            n_aux=n_aux,
            save_path=pop_case_out,
        )

        # Directly run logical tomography from in-memory simulation result.
        run_tomography_from_arrays(
            case_name=case_name,
            A=A,
            time_points=time_points,
            avg_rho_t=avg_rho_t,
            H0_total=ops["H0_total"],
            plot_prefix=f"dual_rail_tomography_{case_name}",
            dpi=200,
            data_label=f"dual_rail.py in-memory result ({case_name})",
            n_plot_points=10,
        )

    out_path = Path(args.output)
    with out_path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved averaged density matrices to: {out_path.resolve()}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
