import numpy as np
import os
os.chdir('/home/yunwei/SNAIL/transmon/monte_carlo/new/full_simulation')
import qutip as qt
from noise_generator import GenerateNoise
from hamiltonian_generator import Hamiltonian
from system import *
from joblib import Parallel, delayed
import pickle


class Simulation:
    def __init__(self, A, initial_state, t_max, sample_rate, num_realizations, S0):
        self.A = A
        self.initial_state = initial_state

        self.phi_ex = 0.2
        self.Ej = 30.19
        self.Ec = 0.1

        self.sc = Hamiltonian(self.phi_ex, self.Ej, self.Ec, [5, 3])
        self.optimal_omega, rate = self.sc.optimal_omegad(self.A)
        self.optimal_omega = self.optimal_omega * 2 * np.pi
        self.sc = Hamiltonian(self.phi_ex, self.Ej, self.Ec, [3, 2])
        self.sample_rate = sample_rate
        self.num_realizations = num_realizations
        self.S0 = S0
        self.t_max = t_max
        self.cplus_state, self.kick_and_sigmax, self.get_projector = self.sc.setup_floquet_system(A, self.optimal_omega)
        self.detuning = np.real(self.sc.H)[2, 2] - self.optimal_omega

    def noise_check(self, if_plot=False):
        tnoise_max = int(self.t_max)
        relative_PSD_strength = self.S0**2
        ifwhite = False
        gn = GenerateNoise(self.sample_rate, tnoise_max, relative_PSD_strength, self.num_realizations, ifwhite)
        trajs = gn.generate_colored_noise()
        self.trajs = trajs
        if if_plot:
            gn.analyze_noise_psd(trajs)

    def operators(self):
        sds = self.sc.noise
        sop = self.sc.s
        H_control = self.sc.H_control
        diagonal_energies = np.diag(self.sc.H) - self.sc.H[0, 0]
        H0 = np.diag(diagonal_energies)
        sds = qt.Qobj(sds)
        sop = qt.Qobj(sop)
        H_control = qt.Qobj(H_control)
        H0 = qt.Qobj(H0)
        return sds, sop, H_control, H0


def simulate_trajectory(i, sim, A, t_max, sample_rate, sim_time, initial_state):
    """Single trajectory returning density matrices at each time step."""
    sds, sop, H_control, H0 = sim.operators()
    time_points = np.linspace(0, sim_time, sim_time)
    drive_term = lambda t, args: A * np.cos(sim.optimal_omega * t)
    gamma = 1 / (2e4)
    opts = {'nsteps': sim.t_max, 'atol': 1e-12, 'rtol': 1e-12}
    c_ops = [np.sqrt(gamma) * qt.Qobj(sop)]
    traj = np.cos(sim.phi_ex * np.pi) * (np.cos(sim.trajs[i] * np.pi) - 1) - np.sin(sim.phi_ex * np.pi) * np.sin(sim.trajs[i] * np.pi)
    H = [H0, [sds, lambda t, args: traj[int(t * sample_rate)] if t < t_max else 0], [H_control, drive_term]]
    result = qt.mesolve(H, initial_state, time_points, c_ops, options=opts)

    dim = initial_state.shape[0]
    rho_traj = np.zeros((len(time_points), dim, dim), dtype=complex)
    for j in range(len(time_points)):
        rho_traj[j] = result.states[j].full()
    return rho_traj


if __name__ == '__main__':
    DATA_FILE = 'total_rate_debug_data.pkl'

    A = 1e-4 * 2 * np.pi
    S0_values = [1e-5]
    t1_sim_time = 10000
    t2_sim_time = 50000

    results_all = {}
    for idx, S0 in enumerate(S0_values):
        print(f"[{idx+1}/{len(S0_values)}] S0={S0:.2e}: RUNNING...")

        t_max = 100000
        num_realizations = 100
        sample_rate = 1
        sim = Simulation(A, [], t_max, sample_rate, num_realizations, S0)
        sim.noise_check()
        num_trajectories = len(sim.trajs)

        floquet_states, sigmax, sigmas = sim.kick_and_sigmax(0)
        floquet_ground = floquet_states[:, 0]
        floquet_excited = floquet_states[:, 1]

        # ---- T1: start from excited state ----
        print(f"  Running T1 ({num_trajectories} trajectories, {t1_sim_time} steps)...")
        init_T1 = qt.Qobj(floquet_excited)
        t1_time_points = np.linspace(0, t1_sim_time, t1_sim_time)
        t1_rhos = Parallel(n_jobs=-1)(
            delayed(simulate_trajectory)(i, sim, A, t_max, sample_rate, t1_sim_time, init_T1)
            for i in range(num_trajectories)
        )
        avg_rho_T1 = np.mean(np.array(t1_rhos), axis=0)  # (t1_sim_time, dim, dim)

        # ---- T2: start from superposition state ----
        print(f"  Running T2 ({num_trajectories} trajectories, {t2_sim_time} steps)...")
        init_T2 = (qt.Qobj(floquet_ground) + qt.Qobj(floquet_excited)).unit()
        t2_time_points = np.linspace(0, t2_sim_time, t2_sim_time)
        t2_rhos = Parallel(n_jobs=-1)(
            delayed(simulate_trajectory)(i, sim, A, t_max, sample_rate, t2_sim_time, init_T2)
            for i in range(num_trajectories)
        )
        avg_rho_T2 = np.mean(np.array(t2_rhos), axis=0)  # (t2_sim_time, dim, dim)

        # ---- Transmon decay: start from |10⟩ ----
        floquet_10 = floquet_states[:, 2]
        print(f"  Running transmon decay from |10⟩ ({num_trajectories} trajectories, {t1_sim_time} steps)...")
        init_10 = qt.Qobj(floquet_10)
        t10_time_points = np.linspace(0, t1_sim_time, t1_sim_time)
        t10_rhos = Parallel(n_jobs=-1)(
            delayed(simulate_trajectory)(i, sim, A, t_max, sample_rate, t1_sim_time, init_10)
            for i in range(num_trajectories)
        )
        avg_rho_10 = np.mean(np.array(t10_rhos), axis=0)  # (t1_sim_time, dim, dim)

        results_all[S0] = {
            'avg_rho_T1': avg_rho_T1,
            't1_time_points': t1_time_points,
            'avg_rho_T2': avg_rho_T2,
            't2_time_points': t2_time_points,
            'avg_rho_10': avg_rho_10,
            't10_time_points': t10_time_points,
            'fit_omega': sim.sc.fit_omega,
            'optimal_omega': sim.optimal_omega,
        }
        print(f"  S0={S0:.2e}: DONE")

        with open(DATA_FILE, 'wb') as f:
            pickle.dump({'A': A, 'S0_values': S0_values, 'results': results_all}, f)
        print(f"  -> Saved to {DATA_FILE}")

    print("=== ALL DONE ===")
