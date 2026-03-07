import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import qutip as qt
from joblib import Parallel, delayed
from system import sort_eigenpairs
from noise_generator import GenerateNoise
from hamiltonian_generator import Hamiltonian
from plot_reduced_wigner_cavity import plot_wigner_10points_both


class Simulation:
    def __init__(
        self,
        A,
        initial_state,
        t_max,
        sample_rate,
        num_realizations,
        S0,
        noise_t_max=None,
        noise_dt=1.0,
        solver_nsteps=None,
        noise_eval_t_max=None,
        n_transmon=3,
        n_cavity=5,
    ):
        self.A = A
        self.initial_state = initial_state

        self.phi_ex = 0.2
        self.Ej = 30.19
        self.Ec = 0.1

        self.n_transmon = int(n_transmon)
        self.n_cavity = int(n_cavity)
        self.total_dim = self.n_transmon * self.n_cavity

        self.sc = Hamiltonian(self.phi_ex, self.Ej, self.Ec, [5, 10])
        self.optimal_omega, _ = self.sc.optimal_omegad(self.A)
        self.optimal_omega = self.optimal_omega * 2 * np.pi
        self.sc = Hamiltonian(self.phi_ex, self.Ej, self.Ec, [self.n_transmon, self.n_cavity])

        self.sample_rate = sample_rate
        self.num_realizations = num_realizations
        self.S0 = S0
        self.t_max = t_max
        self.noise_t_max = t_max if noise_t_max is None else noise_t_max
        self.noise_dt = noise_dt
        self.solver_nsteps = t_max if solver_nsteps is None else solver_nsteps
        self.noise_eval_t_max = noise_eval_t_max

        self.cplus_state, self.kick_and_sigmax, self.get_projector = self.sc.setup_floquet_system(
            A, self.optimal_omega
        )
        self.detuning = np.real(self.sc.H)[2, 2] - self.optimal_omega

    def noise_check(self):
        n_noise_samples = int(np.ceil(float(self.noise_t_max) / float(self.noise_dt)))
        if n_noise_samples <= 0:
            raise ValueError("Computed noise sample count must be positive.")
        relative_psd_strength = self.S0**2
        ifwhite = False
        gn = GenerateNoise(
            1, n_noise_samples, relative_psd_strength, self.num_realizations, ifwhite
        )
        self.trajs = gn.generate_colored_noise()
        if self.noise_eval_t_max is not None:
            n_keep = int(np.ceil(float(self.noise_eval_t_max) / float(self.noise_dt)))
            self.trajs = self.trajs[:, :n_keep]

    def operators(self):
        op_dims = [[self.n_transmon, self.n_cavity], [self.n_transmon, self.n_cavity]]
        tol = 1e-12

        a_s_dressed = np.kron(self.sc.annihilation(self.n_transmon), np.eye(self.n_cavity))
        a_c_dressed = np.kron(np.eye(self.n_transmon), self.sc.annihilation(self.n_cavity))
        n_s = np.kron(np.diag(np.arange(self.n_transmon)), np.eye(self.n_cavity))
        n_c = np.kron(np.eye(self.n_transmon), np.diag(np.arange(self.n_cavity)))
        self.a_s_dressed = a_s_dressed
        self.a_c_dressed = a_c_dressed

        sds_raw = np.asarray(self.sc.noise, dtype=complex).copy()
        diag_vals = np.diag(sds_raw) - sds_raw[0, 0]
        np.fill_diagonal(sds_raw, diag_vals)
        diag_mask = np.eye(sds_raw.shape[0], dtype=bool)
        sds_diag = np.zeros_like(sds_raw, dtype=complex)
        sds_diag[diag_mask] = sds_raw[diag_mask]

        H_control_mat = np.zeros_like(np.asarray(self.sc.H_control), dtype=complex)
        for i in range(self.total_dim - self.n_cavity):
            j = i + self.n_cavity
            H_control_mat[i, j] = self.sc.H_control[i, j]
            H_control_mat[j, i] = self.sc.H_control[j, i]

        sop_raw = np.asarray(self.sc.s, dtype=complex)
        s_mask = np.abs(a_s_dressed) > tol
        c_mask = np.abs(a_c_dressed) > tol
        sop_selected = np.zeros_like(sop_raw, dtype=complex)
        cop_selected = np.zeros_like(sop_raw, dtype=complex)
        sop_selected[s_mask] = sop_raw[s_mask]
        cop_selected[c_mask] = sop_raw[c_mask]
        self.sop_selected = sop_selected
        self.cop_selected = cop_selected

        self.s_nonzero_indices = np.argwhere(s_mask)
        self.c_nonzero_indices = np.argwhere(c_mask)
        self.sop_selected_indices = np.argwhere(np.abs(sop_selected) > tol)
        self.cop_selected_indices = np.argwhere(np.abs(cop_selected) > tol)
        self.sds_selected_indices = np.argwhere(np.abs(sds_diag) > tol)

        sds = qt.Qobj(sds_diag, dims=op_dims)
        sop = qt.Qobj(sop_selected, dims=op_dims)
        cop = qt.Qobj(cop_selected, dims=op_dims)
        H0_diag = np.diag(np.diag(self.sc.H) - self.sc.H[0, 0])
        omegac = np.abs(H0_diag[1, 1])
        H0_rot_raw = H0_diag - self.optimal_omega * n_s - omegac * n_c + (self.A / 2.0) * H_control_mat
        evals_rot, U = np.linalg.eigh(H0_rot_raw)
        evals_rot, U = sort_eigenpairs(evals_rot, U)
        evals_rot = np.real(evals_rot - evals_rot[0])
        H0_rot = np.diag(evals_rot)
        omega_c = evals_rot[1]
        H0_rot_raw = H0_rot_raw  - omega_c * n_c

        new_c = U.T.conj() @ cop_selected @ U
        new_s = U.T.conj() @ sop_selected @ U
        total_dim = U.shape[1]
        eig_candidates = [0, 2, 4]
        eig_indices = []
        for c in eig_candidates:
            idx = min(c, total_dim - 1)
            if idx not in eig_indices:
                eig_indices.append(idx)
        while len(eig_indices) < 3:
            eig_indices.append(total_dim - 1)
        eig00, eig02, eig04 = eig_indices[:3]

        ket00 = U[:, eig00].copy()
        ket02 = U[:, eig02].copy()
        ket04 = U[:, eig04].copy()

        ref00 = 0
        ref02 = min(2, self.n_cavity - 1)
        ref04 = min(4, self.n_cavity - 1)
        if np.abs(ket00[ref00]) > 1e-15:
            ket00 = ket00 / ket00[ref00]
        if np.abs(ket02[ref02]) > 1e-15:
            ket02 = ket02 / ket02[ref02]
        if np.abs(ket04[ref04]) > 1e-15:
            ket04 = ket04 / ket04[ref04]

        n00 = np.linalg.norm(ket00)
        n02 = np.linalg.norm(ket02)
        n04 = np.linalg.norm(ket04)
        if n00 > 1e-15:
            ket00 = ket00 / n00
        if n02 > 1e-15:
            ket02 = ket02 / n02
        if n04 > 1e-15:
            ket04 = ket04 / n04

        ket0 = np.sqrt(1 / 2) * (ket00 + ket04)
        ket1 = ket02

        initial_state = np.sqrt(1 / 2) * (ket0 + ket1)


        self.driven_initial_state = qt.Qobj(initial_state, dims=[[self.n_transmon, self.n_cavity], [1, 1]])

        self.omega_c_from_H0 = omega_c
        self.U_rot = U

        H0_rot_raw_qobj = qt.Qobj(H0_rot_raw, dims=op_dims)
        return sds, sop, cop, H0_rot_raw_qobj


def simulate_single_trajectory(
    traj_idx, raw_trajs, noise_dt, solver_nsteps,
    sds, sop, cop, H0_rot_raw, psi0, time_points,
):
    """Run mesolve for one noise trajectory. Operators are passed in (not rebuilt)."""
    gamma = 1 / (2e4)
    c_ops = [np.sqrt(gamma) * sop, np.sqrt(gamma) * cop]
    opts = {
        "nsteps": int(solver_nsteps),
        "atol": 1e-12,
        "rtol": 1e-12,
        "progress_bar": False,
    }

    # Time-dependent term: (d omega_b / d Phi) * deltaPhi(t) with deltaPhi from trajectory.
    traj = np.asarray(raw_trajs[traj_idx], dtype=complex)

    # Array-based coefficient: no Python callback at each ODE step.
    n_samp = len(traj)
    t_noise = np.arange(n_samp, dtype=float) * noise_dt
    noise_qevo = qt.coefficient(np.asarray(traj, dtype=complex), tlist=t_noise)
    H = qt.QobjEvo([H0_rot_raw, [sds, noise_qevo]])

    result = qt.mesolve(H, psi0, time_points, c_ops, options=opts)
    rho_t = [st if st.isoper else qt.ket2dm(st) for st in result.states]
    return np.stack([rho.full() for rho in rho_t], axis=0)


def ptrace_cavity_vectorized(avg_rho_tc, n_transmon=3, n_cavity=5):
    """Partial trace over transmon via numpy reshaping — no Qobj loop."""
    n_t = avg_rho_tc.shape[0]
    reshaped = avg_rho_tc.reshape(n_t, n_transmon, n_cavity, n_transmon, n_cavity)
    return np.trace(reshaped, axis1=1, axis2=3)


def run_average_density_matrix(
    A=0.0,
    t_max=100000,
    sim_time=2000,
    num_time_points=None,
    num_realizations=100,
    S0=1e-5,
    sample_rate=1,
    noise_dt=1.0,
    n_jobs=-1,
    noise_t_max=None,
    solver_nsteps=None,
    n_transmon=3,
    n_cavity=5,
    max_save_points=100,
    save_sampled_only=False,
):
    if num_time_points is None:
        num_time_points = max(2, int(np.ceil(sim_time) / 10))
    if save_sampled_only:
        num_time_points = min(int(num_time_points), int(max_save_points))
        num_time_points = max(2, num_time_points)

    noise_eval_t_max = sim_time
    sim = Simulation(
        A, [], t_max, sample_rate, num_realizations, S0,
        noise_t_max=noise_t_max,
        noise_dt=noise_dt,
        solver_nsteps=solver_nsteps,
        noise_eval_t_max=noise_eval_t_max,
        n_transmon=n_transmon,
        n_cavity=n_cavity,
    )
    sim.noise_check()
    sds, sop, cop, H0_rot_raw = sim.operators()
    if not np.isclose(A, 0.0):
        psi0 = sim.driven_initial_state
    else:
        ket0 = qt.Qobj(sim.kick_and_sigmax(0)[0][:, 0], dims=[[3, 5], [1, 1]])
        ket2 = qt.Qobj(sim.kick_and_sigmax(0)[0][:, 2], dims=[[3, 5], [1, 1]])
        ket4 = qt.Qobj(sim.kick_and_sigmax(0)[0][:, 4], dims=[[3, 5], [1, 1]])
        ket0L = (ket0 + ket4).unit()
        ket1L = ket2
        psi0 = (ket0L + ket1L).unit()
    time_points = np.linspace(0, sim_time, num_time_points)

    # Opt #1: build operators once
    operator_elements = {
        "n_transmon": sim.n_transmon,
        "n_cavity": sim.n_cavity,
        "omega_c": sim.omega_c_from_H0,
        "H0_rot_raw": H0_rot_raw.full(),
        "sds_selected_nonzero": [
            (int(i), int(j), complex(sds[i, j]))
            for i, j in sim.sds_selected_indices
        ],
        "s_nonzero": [
            (int(i), int(j), complex(sim.a_s_dressed[i, j]))
            for i, j in sim.s_nonzero_indices
        ],
        "c_nonzero": [
            (int(i), int(j), complex(sim.a_c_dressed[i, j]))
            for i, j in sim.c_nonzero_indices
        ],
        "sop_selected_nonzero": [
            (int(i), int(j), complex(sim.sop_selected[i, j]))
            for i, j in sim.sop_selected_indices
        ],
        "cop_selected_nonzero": [
            (int(i), int(j), complex(sim.cop_selected[i, j]))
            for i, j in sim.cop_selected_indices
        ],
    }

    # Opt #3: running sum instead of storing all trajectory arrays
    n_traj = len(sim.trajs)
    rho_sum = None

    results = Parallel(n_jobs=n_jobs)(
        delayed(simulate_single_trajectory)(
            i, sim.trajs, sim.noise_dt, sim.solver_nsteps, sds, sop, cop, H0_rot_raw, psi0, time_points,
        )
        for i in range(n_traj)
    )
    for rho_arr in results:
        if rho_sum is None:
            rho_sum = rho_arr.copy()
        else:
            rho_sum += rho_arr

    avg_rho_tc = rho_sum / n_traj

    # Opt #4: vectorized partial trace
    avg_rho_cav = ptrace_cavity_vectorized(avg_rho_tc, n_transmon=sim.n_transmon, n_cavity=sim.n_cavity)

    return time_points, avg_rho_tc, avg_rho_cav, operator_elements


def downsample_for_saving(time_points, avg_rho_tc, avg_rho_cav, max_save_points=100):
    n_t = len(time_points)
    if n_t <= max_save_points:
        return time_points, avg_rho_tc, avg_rho_cav
    save_idx = np.linspace(0, n_t - 1, max_save_points, dtype=int)
    return time_points[save_idx], avg_rho_tc[save_idx], avg_rho_cav[save_idx]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run undriven/driven transmon x cavity simulations and save averaged density matrices "
            "with downsampled time points."
        )
    )
    parser.add_argument("--t-max", type=int, default=100000, help="Noise trajectory length.")
    parser.add_argument("--sim-time", type=float, default=300000.0, help="Simulation end time.")
    parser.add_argument(
        "--num-time-points",
        type=int,
        default=None,
        help="Number of simulation time points before save-downsampling (default: ceil(sim_time)).",
    )
    parser.add_argument("--num-realizations", type=int, default=100, help="Number of trajectories.")
    parser.add_argument("--S0", type=float, default=1e-5, help="Noise amplitude.")
    parser.add_argument("--sample-rate", type=int, default=1, help="Samples per ns.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs.")
    parser.add_argument(
        "--driven-A-over-pi",
        type=float,
        default=10e-3,
        help="Driven amplitude specified as A/pi.",
    )
    parser.add_argument(
        "--driven-noise-t-max",
        type=float,
        default=1e5,
        help="Driven trajectory-generation t_max.",
    )
    parser.add_argument(
        "--driven-dt",
        type=float,
        default=1.0,
        help="Driven trajectory time step dt; sample rate is 1/dt.",
    )
    parser.add_argument(
        "--undriven-noise-t-max",
        type=float,
        default=1e8,
        help="Undriven trajectory-generation t_max.",
    )
    parser.add_argument(
        "--undriven-dt",
        type=float,
        default=1e3,
        help="Undriven trajectory time step dt; sample rate is 1/dt.",
    )
    parser.add_argument(
        "--max-save-points",
        type=int,
        default=100,
        help="Maximum number of averaged density matrices to save over time.",
    )
    parser.add_argument(
        "--save-sampled-only",
        action="store_true",
        help="Simulate only on sampled save points to reduce memory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="avg_density_matrix_over_time_undriven_driven.pkl",
        help="Output pickle path.",
    )
    parser.add_argument(
        "--print-h0-rot-raw",
        action="store_true",
        help="Print H0_rot_raw matrix for each run case.",
    )
    parser.add_argument(
        "--plot-save-path",
        type=str,
        default=None,
        help="Optional output path for Wigner plot PNG (default: timestamped file).",
    )
    parser.add_argument("--plot-xlim", type=float, default=6.0, help="Wigner axis limit.")
    parser.add_argument("--plot-grid", type=int, default=201, help="Wigner grid size.")
    parser.add_argument("--plot-dpi", type=int, default=200, help="DPI for Wigner plot image.")
    parser.add_argument("--n-transmon", type=int, default=3, help="Transmon Hilbert-space truncation.")
    parser.add_argument("--n-cavity", type=int, default=5, help="Cavity Hilbert-space truncation.")
    args = parser.parse_args()

    if args.undriven_dt <= 0:
        raise ValueError("undriven_dt must be > 0")
    if args.driven_dt <= 0:
        raise ValueError("driven_dt must be > 0")

    run_cases = {
        "undriven": {
            "A": 0.0 * 2 * np.pi,
            "noise_t_max": args.undriven_noise_t_max,
            "noise_dt": args.undriven_dt,
            "solver_nsteps": args.t_max,
        },
        "driven": {
            "A": args.driven_A_over_pi * np.pi,
            "noise_t_max": args.driven_noise_t_max,
            "noise_dt": args.driven_dt,
            "solver_nsteps": args.t_max,
        },
    }

    results = {}
    for case_name, cfg in run_cases.items():
        time_points, avg_rho_tc, avg_rho_cav, operator_elements = run_average_density_matrix(
            A=cfg["A"],
            t_max=args.t_max,
            sim_time=args.sim_time,
            num_time_points=args.num_time_points,
            num_realizations=args.num_realizations,
            S0=args.S0,
            sample_rate=args.sample_rate,
            noise_dt=cfg["noise_dt"],
            n_jobs=args.n_jobs,
            noise_t_max=cfg["noise_t_max"],
            solver_nsteps=cfg["solver_nsteps"],
            n_transmon=args.n_transmon,
            n_cavity=args.n_cavity,
            max_save_points=args.max_save_points,
            save_sampled_only=args.save_sampled_only,
        )
        if args.print_h0_rot_raw:
            print(f"{case_name} H0_rot_raw:")
            print(np.array2string(operator_elements["H0_rot_raw"], precision=8, suppress_small=False))

        t_save, rho_tc_save, rho_cav_save = downsample_for_saving(
            time_points, avg_rho_tc, avg_rho_cav, max_save_points=args.max_save_points
        )
        results[case_name] = {
            "A": cfg["A"],
            "noise_t_max": cfg["noise_t_max"],
            "noise_dt": cfg["noise_dt"],
            "time_points": t_save,
            "avg_rho_tc": rho_tc_save,
            "avg_rho_cav": rho_cav_save,
        }

    payload = {
        "undriven": results["undriven"],
        "driven": results["driven"],
        "params": vars(args),
    }
    output_path = Path(args.output)
    with output_path.open("wb") as f:
        pickle.dump(payload, f)

    if args.n_cavity != 5:
        raise ValueError("Wigner plotting currently supports n-cavity=5 only.")
    if args.plot_save_path is None:
        plot_save_path = (
            Path(__file__).resolve().parent
            / f"cavity_wigner_10points_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
    else:
        plot_save_path = args.plot_save_path

    undriven_rho_cav_plot = ptrace_cavity_vectorized(
        np.asarray(results["undriven"]["avg_rho_tc"]),
        n_transmon=args.n_transmon,
        n_cavity=args.n_cavity,
    )
    driven_rho_cav_plot = ptrace_cavity_vectorized(
        np.asarray(results["driven"]["avg_rho_tc"]),
        n_transmon=args.n_transmon,
        n_cavity=args.n_cavity,
    )
    plot_wigner_10points_both(
        undriven_data=(results["undriven"]["time_points"], undriven_rho_cav_plot),
        driven_data=(results["driven"]["time_points"], driven_rho_cav_plot),
        save_path=plot_save_path,
        xlim=args.plot_xlim,
        grid=args.plot_grid,
        dpi=args.plot_dpi,
    )


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
