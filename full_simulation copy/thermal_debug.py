import numpy as np
import os
os.chdir('/home/yunwei/SNAIL/transmon/monte_carlo/new/full_simulation')
import matplotlib.pyplot as plt
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
    DATA_FILE = 'thermal_debug_data.pkl'

    A = 1e-4 * 2 * np.pi
    S0 = 1e-5
    sim_time = 10000

    t_max = 100000
    num_realizations = 100
    sample_rate = 1
    sim = Simulation(A, [], t_max, sample_rate, num_realizations, S0)
    sim.noise_check()
    num_trajectories = len(sim.trajs)

    floquet_states, sigmax, sigmas = sim.kick_and_sigmax(0)
    floquet_ground = floquet_states[:, 0]
    init_state = qt.Qobj(floquet_ground)

    print(f"Running {num_trajectories} trajectories, {sim_time} steps, ground state initial...")
    rhos = Parallel(n_jobs=-1)(
        delayed(simulate_trajectory)(i, sim, A, t_max, sample_rate, sim_time, init_state)
        for i in range(num_trajectories)
    )
    avg_rho = np.mean(np.array(rhos), axis=0)
    time_points = np.linspace(0, sim_time, sim_time)

    with open(DATA_FILE, 'wb') as f:
        pickle.dump({'A': A, 'S0': S0, 'avg_rho': avg_rho, 'time_points': time_points}, f)
    print(f"Saved to {DATA_FILE}")

    # Compute P[0] and P[1] from avg density matrix
    get_projector = sim.get_projector
    kick_and_sigmax = sim.kick_and_sigmax
    P0 = np.zeros(len(time_points))
    P1 = np.zeros(len(time_points))
    for j, t in enumerate(time_points):
        fs, _, _ = kick_and_sigmax(t)
        projs = get_projector(fs)
        rho_j = qt.Qobj(avg_rho[j])
        P0[j] = np.real(qt.expect(projs[0], rho_j))
        P1[j] = np.real(qt.expect(projs[1], rho_j))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_points, P0, label='P[0] (ground)')
    ax.plot(time_points, P1, label='P[1] (excited)')
    ax.set_xlabel('Time (ns)', fontsize=14)
    ax.set_ylabel('Population', fontsize=14)
    ax.set_title(f'Ground state initial, A={A/(2*np.pi)*1e3:.2f} mHz, S0={S0:.1e}', fontsize=13)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig('thermal_debug_plot.pdf')
    print("Plot saved to thermal_debug_plot.pdf")
    plt.show()
