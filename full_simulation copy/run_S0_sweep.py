import numpy as np
import os
os.chdir('/home/yunwei/SNAIL/transmon/monte_carlo/new/full_simulation')

import pickle
import warnings
from datetime import datetime
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
from noise_generator import GenerateNoise
from hamiltonian_generator import Hamiltonian
from system import *
import qutip as qt
import matplotlib.pyplot as plt

LOG_FILE = '/home/yunwei/SNAIL/transmon/monte_carlo/new/full_simulation/S0_sweep_progress.log'
DATA_FILE = '/home/yunwei/SNAIL/transmon/monte_carlo/new/full_simulation/S0_sweep_raw_data.pkl'
RESULT_FILE = '/home/yunwei/SNAIL/transmon/monte_carlo/new/full_simulation/S0_sweep_results.pkl'

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')
        f.flush()

# ── Simulation class (same as notebook) ──
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

    def operators(self):
        sds = self.sc.noise
        sop = self.sc.s
        H_control = self.sc.H_control
        H0 = self.sc.H
        diagonal_energies = np.diag(self.sc.H) - self.sc.H[0, 0]
        H0 = np.diag(diagonal_energies)
        sds = qt.Qobj(sds)
        sop = qt.Qobj(sop)
        H_control = qt.Qobj(H_control)
        H0 = qt.Qobj(H0)
        return sds, sop, H_control, H0

# ── Fitting functions ──
def fit_T1_function(time_points, population_data):
    def fit1(t, T1):
        return 1 - np.exp(-t / T1)
    try:
        popt, _ = curve_fit(fit1, time_points, population_data, p0=[np.median(time_points)])
        return popt[0]
    except (RuntimeError, ValueError):
        return np.nan

def fit_T2_function(time_points, population_data, omega):
    def T2fit(t, T2):
        return np.exp(-t / T2) * np.cos(omega * t)
    try:
        popt, _ = curve_fit(T2fit, time_points, population_data, p0=[1e6])
        return popt[0]
    except (RuntimeError, ValueError):
        return np.nan

def calculate_t_phi_uncertainty_via_bootstrap(
    t1_population_matrix, t1_time_points, fit_t1_func,
    t2_population_matrix, t2_time_points, fit_t2_func,
    t2_omega, num_bootstrap_samples=1000, n_jobs=-1
):
    num_t1_traj = t1_population_matrix.shape[0]
    num_t2_traj = t2_population_matrix.shape[0]

    def bootstrap_single(seed):
        np.random.seed(seed)
        idx1 = np.random.randint(0, num_t1_traj, size=num_t1_traj)
        avg_t1 = np.mean(t1_population_matrix[idx1], axis=0)
        T1 = fit_t1_func(t1_time_points, avg_t1)
        idx2 = np.random.randint(0, num_t2_traj, size=num_t2_traj)
        avg_t2 = np.mean(t2_population_matrix[idx2], axis=0)
        T2 = fit_t2_func(t2_time_points, avg_t2, t2_omega)
        if np.isnan(T1) or np.isnan(T2) or T1 <= 0 or T2 <= 0:
            return np.nan
        return 1 / T2 - 1 / (2 * T1)

    results = Parallel(n_jobs=n_jobs)(
        delayed(bootstrap_single)(i) for i in range(num_bootstrap_samples)
    )
    valid = [r for r in results if not np.isnan(r)]
    if not valid:
        return np.nan, np.nan
    return np.mean(valid), np.std(valid, ddof=1)

# ── Full simulation ──
def run_full_simulation(A, S0):
    initial_state = []
    t_max = 100000
    num_realizations = 100
    sample_rate = 1
    sim = Simulation(A, initial_state, t_max, sample_rate, num_realizations, S0)
    sim.noise_check()

    def simulate_trajectory_T1(i, sim, A, t_max, sample_rate, sim_time):
        sds, sop, H_control, H0 = sim.operators()
        time_points = np.linspace(0, sim_time, sim_time)
        floquet_states, sigmax, sigmas = sim.kick_and_sigmax(0)
        get_projector = sim.get_projector
        floquet_excited = floquet_states[:, 1]
        initial_state = qt.Qobj(floquet_excited)
        drive_term = lambda t, args: A * np.cos(sim.optimal_omega * t)
        gamma = 1 / (2e4)
        opts = {'nsteps': sim.t_max, 'atol': 1e-12, 'rtol': 1e-12}
        c_ops = [np.sqrt(gamma) * qt.Qobj(sop)]
        traj = np.cos(sim.phi_ex * np.pi) * (np.cos(sim.trajs[i] * np.pi) - 1) - np.sin(sim.phi_ex * np.pi) * np.sin(sim.trajs[i] * np.pi)
        H = [H0, [sds, lambda t, args: traj[int(t * sample_rate)] if t < t_max else 0], [H_control, drive_term]]
        result = qt.mesolve(H, initial_state, time_points, c_ops, options=opts)
        avg_values_single = np.zeros(len(time_points))
        for j, t in enumerate(time_points):
            floquet_states, sigmax, sigmas = sim.kick_and_sigmax(t)
            projectors = get_projector(floquet_states)
            P00 = projectors[0]
            avg_values_single[j] = qt.expect(P00, result.states[j])
        return avg_values_single

    t1_sim_time = 10000
    num_trajectories = len(sim.trajs)
    t1_time_points = np.linspace(0, t1_sim_time, t1_sim_time)
    avg_values = Parallel(n_jobs=-1)(
        delayed(simulate_trajectory_T1)(i, sim, A, t_max, sample_rate, t1_sim_time)
        for i in range(num_trajectories)
    )
    avg_values_T1 = np.array(avg_values)

    def simulate_trajectory_T2(i, sim, A, t_max, sample_rate, sim_time):
        sds, sop, H_control, H0 = sim.operators()
        time_points = np.linspace(0, sim_time, sim_time)
        floquet_states, sigmax, sigmas = sim.kick_and_sigmax(0)
        get_projector = sim.get_projector
        floquet_ground = floquet_states[:, 0]
        floquet_excited = floquet_states[:, 1]
        initial_state = (qt.Qobj(floquet_ground) + qt.Qobj(floquet_excited)).unit()
        drive_term = lambda t, args: A * np.cos(sim.optimal_omega * t)
        gamma = 1 / (2e4)
        opts = {'nsteps': sim.t_max, 'atol': 1e-12, 'rtol': 1e-12}
        c_ops = [np.sqrt(gamma) * qt.Qobj(sop)]
        traj = np.cos(sim.phi_ex * np.pi) * (np.cos(sim.trajs[i] * np.pi) - 1) - np.sin(sim.phi_ex * np.pi) * np.sin(sim.trajs[i] * np.pi)
        H = [H0, [sds, lambda t, args: traj[int(t * sample_rate)] if t < t_max else 0], [H_control, drive_term]]
        result = qt.mesolve(H, initial_state, time_points, c_ops, options=opts)
        avg_values_single = np.zeros(len(time_points))
        for j, t in enumerate(time_points):
            floquet_states, sigmax, sigmas = sim.kick_and_sigmax(t)
            avg_values_single[j] = qt.expect(qt.Qobj(sigmax), result.states[j])
        return avg_values_single

    t2_sim_time = 50000
    num_trajectories = len(sim.trajs)
    t2_time_points = np.linspace(0, t2_sim_time, t2_sim_time)
    avg_values = Parallel(n_jobs=-1)(
        delayed(simulate_trajectory_T2)(i, sim, A, t_max, sample_rate, t2_sim_time)
        for i in range(num_trajectories)
    )
    avg_values_T2 = np.array(avg_values)

    return avg_values_T1, t1_time_points, avg_values_T2, t2_time_points, sim


# ── Main: fixed A, sweep S0 ──
A = 3e-3 * 2 * np.pi
S0_values = np.logspace(-6, -5, 10)

try:
    with open(DATA_FILE, 'rb') as f:
        raw_data = pickle.load(f)
    log(f"Loaded existing raw data from {DATA_FILE}")
except (FileNotFoundError, EOFError):
    raw_data = {}
    log("Starting fresh")

log(f"Fixed A = 3.0 mHz, sweeping {len(S0_values)} S0 values")

for idx, S0 in enumerate(S0_values):
    key = (A, S0)
    if key in raw_data:
        log(f"[{idx+1}/{len(S0_values)}] S0={S0:.2e}: SKIPPED (already done)")
        continue

    log(f"[{idx+1}/{len(S0_values)}] S0={S0:.2e}: RUNNING T1 + T2 simulation...")
    avg_values_T1, t1_time_points, avg_values_T2, t2_time_points, sim = run_full_simulation(A, S0)

    raw_data[key] = {
        'avg_values_T1': avg_values_T1,
        't1_time_points': t1_time_points,
        'avg_values_T2': avg_values_T2,
        't2_time_points': t2_time_points,
        'fit_omega': sim.sc.fit_omega,
    }
    log(f"[{idx+1}/{len(S0_values)}] S0={S0:.2e}: DONE (T1 shape={avg_values_T1.shape}, T2 shape={avg_values_T2.shape})")

    with open(DATA_FILE, 'wb') as f:
        pickle.dump(raw_data, f)
    log(f"  -> Saved raw data to {DATA_FILE}")

log("=== ALL SIMULATIONS DONE ===")

# ── Bootstrap ──
log("Starting bootstrap...")
results = {'avg': [], 'std': []}
for S0 in S0_values:
    key = (A, S0)
    d = raw_data[key]
    avg, std = calculate_t_phi_uncertainty_via_bootstrap(
        d['avg_values_T1'], d['t1_time_points'], fit_T1_function,
        d['avg_values_T2'], d['t2_time_points'], fit_T2_function,
        d['fit_omega']
    )
    results['avg'].append(avg)
    results['std'].append(std)
    log(f"S0={S0:.2e}: dephasing_rate avg={avg:.6e}, std={std:.6e}")

# ── Save results ──
output = {
    'A': A,
    'S0_values': S0_values,
    'results': results,
}
with open(RESULT_FILE, 'wb') as f:
    pickle.dump(output, f)
log(f"Bootstrap results saved to {RESULT_FILE}")
