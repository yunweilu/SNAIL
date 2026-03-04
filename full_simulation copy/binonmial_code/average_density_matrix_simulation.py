import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import qutip as qt
from joblib import Parallel, delayed

from noise_generator import GenerateNoise
from hamiltonian_generator import Hamiltonian


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
        solver_nsteps=None,
    ):
        self.A = A
        self.initial_state = initial_state

        self.phi_ex = 0.2
        self.Ej = 30.19
        self.Ec = 0.1

        # Keep the same optimal-frequency flow as notebook code.
        self.sc = Hamiltonian(self.phi_ex, self.Ej, self.Ec, [5, 10])
        self.optimal_omega, _ = self.sc.optimal_omegad(self.A)
        self.optimal_omega = self.optimal_omega * 2 * np.pi
        self.sc = Hamiltonian(self.phi_ex, self.Ej, self.Ec, [3, 5])

        self.sample_rate = sample_rate
        self.num_realizations = num_realizations
        self.S0 = S0
        self.t_max = t_max
        self.noise_t_max = t_max if noise_t_max is None else noise_t_max
        self.solver_nsteps = t_max if solver_nsteps is None else solver_nsteps

        self.cplus_state, self.kick_and_sigmax, self.get_projector = self.sc.setup_floquet_system(
            A, self.optimal_omega
        )
        self.detuning = np.real(self.sc.H)[2, 2] - self.optimal_omega

    def noise_check(self):
        tnoise_max = int(self.noise_t_max)
        relative_psd_strength = self.S0**2
        ifwhite = False
        gn = GenerateNoise(
            self.sample_rate, tnoise_max, relative_psd_strength, self.num_realizations, ifwhite
        )
        self.trajs = gn.generate_colored_noise()

    def operators(self):
        op_dims = [[3, 5], [3, 5]]
        sds = qt.Qobj(self.sc.noise, dims=op_dims)
        sop = qt.Qobj(self.sc.s, dims=op_dims)
        H_control = qt.Qobj(self.sc.H_control, dims=op_dims)
        diagonal_energies = np.diag(self.sc.H) - self.sc.H[0, 0]
        H0 = qt.Qobj(np.diag(diagonal_energies), dims=op_dims)
        return sds, sop, H_control, H0


def build_logical_superposition(sim):
    """Build (|0_L> + |1_L>)/sqrt(2) in transmon(3) ⊗ cavity(5)."""
    ket0 = qt.Qobj(sim.kick_and_sigmax(0)[0][:, 0], dims=[[3, 5], [1, 1]])
    ket2 = qt.Qobj(sim.kick_and_sigmax(0)[0][:, 2], dims=[[3, 5], [1, 1]])
    ket4 = qt.Qobj(sim.kick_and_sigmax(0)[0][:, 4], dims=[[3, 5], [1, 1]])
    ket0L = (ket0 + ket4).unit()
    ket1L = ket2
    return (ket0L + ket1L).unit()


def simulate_single_trajectory(traj_idx, sim, psi0, time_points, gamma):
    sds, sop, H_control, H0 = sim.operators()
    c_ops = [np.sqrt(gamma) * sop]
    opts = {
        "nsteps": int(sim.solver_nsteps),
        "atol": 1e-12,
        "rtol": 1e-12,
        "progress_bar": False,
    }

    traj = (
        np.cos(sim.phi_ex * np.pi) * (np.cos(sim.trajs[traj_idx] * np.pi) - 1)
        - np.sin(sim.phi_ex * np.pi) * np.sin(sim.trajs[traj_idx] * np.pi)
    )
    noise_term = lambda t, args: traj[int(t * sim.sample_rate)] if t < sim.noise_t_max else 0
    drive_term = lambda t, args: sim.A * np.cos(sim.optimal_omega * t)
    H = [H0, [sds, noise_term], [H_control, drive_term]]

    result = qt.mesolve(H, psi0, time_points, c_ops, options=opts)
    rho_t = [st if st.isoper else qt.ket2dm(st) for st in result.states]
    return np.stack([rho.full() for rho in rho_t], axis=0)


def run_average_density_matrix(
    A=0.0,
    t_max=100000,
    sim_time=2000,
    num_time_points=10,
    num_realizations=100,
    S0=1e-5,
    sample_rate=1,
    gamma=1 / (2e4),
    n_jobs=-1,
    noise_t_max=None,
    solver_nsteps=None,
):
    sim = Simulation(
        A,
        [],
        t_max,
        sample_rate,
        num_realizations,
        S0,
        noise_t_max=noise_t_max,
        solver_nsteps=solver_nsteps,
    )
    sim.noise_check()
    psi0 = build_logical_superposition(sim)
    time_points = np.linspace(0, sim_time, num_time_points)

    rho_traj = Parallel(n_jobs=n_jobs)(
        delayed(simulate_single_trajectory)(i, sim, psi0, time_points, gamma)
        for i in range(len(sim.trajs))
    )
    rho_traj = np.asarray(rho_traj)  # (n_traj, n_t, 15, 15)

    avg_rho_tc = np.mean(rho_traj, axis=0)  # (n_t, 15, 15)
    avg_rho_cav = []
    for k in range(avg_rho_tc.shape[0]):
        rho_tc_k = qt.Qobj(avg_rho_tc[k], dims=[[3, 5], [3, 5]])
        avg_rho_cav.append(rho_tc_k.ptrace(1).full())  # trace out transmon
    avg_rho_cav = np.asarray(avg_rho_cav)  # (n_t, 5, 5)

    return time_points, avg_rho_tc, avg_rho_cav


def downsample_for_saving(time_points, avg_rho_tc, avg_rho_cav, max_save_points=1000):
    n_t = len(time_points)
    if n_t <= max_save_points:
        return time_points, avg_rho_tc, avg_rho_cav
    save_idx = np.linspace(0, n_t - 1, max_save_points, dtype=int)
    return time_points[save_idx], avg_rho_tc[save_idx], avg_rho_cav[save_idx]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run undriven/driven transmon⊗cavity simulations and save averaged density matrices "
            "with downsampled time points."
        )
    )
    parser.add_argument("--t-max", type=int, default=100000, help="Noise trajectory length.")
    parser.add_argument("--sim-time", type=float, default=100.0, help="Simulation end time.")
    parser.add_argument(
        "--num-time-points",
        type=int,
        default=5000,
        help="Number of simulation time points before save-downsampling.",
    )
    parser.add_argument("--num-realizations", type=int, default=100, help="Number of trajectories.")
    parser.add_argument("--S0", type=float, default=1e-5, help="Noise amplitude.")
    parser.add_argument("--sample-rate", type=int, default=1, help="Samples per ns.")
    parser.add_argument("--gamma", type=float, default=1 / (2e4), help="Collapse rate.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs.")
    parser.add_argument(
        "--driven-A-over-pi",
        type=float,
        default=10e-3,
        help="Driven amplitude specified as A/pi.",
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
        default=0.01,
        help="Undriven trajectory time step dt; sample rate is 1/dt.",
    )
    parser.add_argument(
        "--max-save-points",
        type=int,
        default=1000,
        help="Maximum number of averaged density matrices to save over time.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="avg_density_matrix_over_time_undriven_driven.pkl",
        help="Output pickle path.",
    )
    args = parser.parse_args()

    if args.undriven_dt <= 0:
        raise ValueError("undriven_dt must be > 0")
    undriven_sample_rate = int(round(1.0 / args.undriven_dt))
    if undriven_sample_rate <= 0:
        raise ValueError("Computed undriven sample rate must be positive")

    run_cases = {
        "undriven": {
            "A": 0.0 * 2 * np.pi,  # A/(2pi)=0
            "noise_t_max": args.undriven_noise_t_max,
            "sample_rate": undriven_sample_rate,
            "solver_nsteps": args.t_max,
        },
        "driven": {
            "A": args.driven_A_over_pi * np.pi,  # A/pi=10e-3 by default
            "noise_t_max": args.t_max,
            "sample_rate": args.sample_rate,
            "solver_nsteps": args.t_max,
        },
    }

    results = {}
    for case_name, cfg in run_cases.items():
        print(
            f"Running {case_name}: A={cfg['A']:.6e}, noise_t_max={cfg['noise_t_max']}, "
            f"sample_rate={cfg['sample_rate']}"
        )
        time_points, avg_rho_tc, avg_rho_cav = run_average_density_matrix(
            A=cfg["A"],
            t_max=args.t_max,
            sim_time=args.sim_time,
            num_time_points=args.num_time_points,
            num_realizations=args.num_realizations,
            S0=args.S0,
            sample_rate=cfg["sample_rate"],
            gamma=args.gamma,
            n_jobs=args.n_jobs,
            noise_t_max=cfg["noise_t_max"],
            solver_nsteps=cfg["solver_nsteps"],
        )
        t_save, rho_tc_save, rho_cav_save = downsample_for_saving(
            time_points, avg_rho_tc, avg_rho_cav, max_save_points=args.max_save_points
        )
        results[case_name] = {
            "A": cfg["A"],
            "noise_t_max": cfg["noise_t_max"],
            "sample_rate": cfg["sample_rate"],
            "time_points": t_save,
            "avg_rho_tc": rho_tc_save,
            "avg_rho_cav": rho_cav_save,
        }
        print(
            f"{case_name}: simulated {len(time_points)} points, saved {len(t_save)} points; "
            f"avg_rho_tc shape {rho_tc_save.shape}, avg_rho_cav shape {rho_cav_save.shape}"
        )

    payload = {
        "undriven": results["undriven"],
        "driven": results["driven"],
        "params": vars(args),
    }
    output_path = Path(args.output)
    with output_path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"Saved undriven/driven averaged density matrices to: {output_path.resolve()}")


if __name__ == "__main__":
    # Keep behavior consistent when run from anywhere.
    os.chdir(Path(__file__).resolve().parent)
    main()
