"""
Test T1 (1→0 transition) using the exact same code as full_simulation_T1.ipynb.
Single run: A = 10e-3 * 2π, S0 = 1e-5, then bootstrap to get T1 with uncertainty.
"""
import os
import sys
import time

import numpy as np

os.chdir('/home/yunwei/SNAIL/full_simulation copy')

import matplotlib.pyplot as plt
from scipy import stats
import colorednoise as cn
import qutip as qt
from noise_generator import GenerateNoise
from hamiltonian_generator import Hamiltonian
from system import *
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
import warnings


class Simulation:
    def __init__(self, A, initial_state, t_max, sample_rate, num_realizations, S0):
        self.A = A
        self.initial_state = initial_state

        self.phi_ex = 0.2
        self.Ej = 30.19
        self.Ec = 0.1

        self.sc = Hamiltonian(self.phi_ex, self.Ej, self.Ec, [5, 10])
        self.optimal_omega, rate = self.sc.optimal_omegad(self.A)
        self.optimal_omega = self.optimal_omega * 2 * np.pi
        self.transmon_levels = 3
        self.cavity_levels = 5
        self.sc = Hamiltonian(self.phi_ex, self.Ej, self.Ec, [self.transmon_levels, self.cavity_levels])
        self.sample_rate = sample_rate
        self.num_realizations = num_realizations
        self.S0 = S0
        self.t_max = t_max
        self.cplus_state, self.kick_and_sigmax, self.get_projector = (
            self.sc.setup_floquet_system(A, self.optimal_omega)
        )
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
        H0 = self.sc.H
        diagonal_energies = np.diag(self.sc.H) - self.sc.H[0, 0]
        H0 = np.diag(diagonal_energies)
        sds = qt.Qobj(sds)
        sop = qt.Qobj(sop)
        H_control = qt.Qobj(H_control)
        H0 = qt.Qobj(H0)
        return sds, sop, H_control, H0


def simulate_trajectory_T1(i, sim, A, t_max, sample_rate, sim_time, initial_state_idx, cavity_idx):
    sds, sop, H_control, H0 = sim.operators()
    time_points = np.linspace(0, sim_time, sim_time)
    floquet_states, sigmax, sigmas = sim.kick_and_sigmax(0)
    get_projector = sim.get_projector
    floquet_state = floquet_states[:, initial_state_idx]
    initial_state = qt.Qobj(floquet_state)
    drive_term = lambda t, args: A * np.cos(sim.optimal_omega * t)
    gamma = 1 / (2e4)
    opts = {'nsteps': sim.t_max, 'atol': 1e-9, 'rtol': 1e-9}
    c_ops = [np.sqrt(gamma) * qt.Qobj(sop)]
    traj = (
        np.cos(sim.phi_ex * np.pi) * (np.cos(sim.trajs[i] * np.pi) - 1)
        - np.sin(sim.phi_ex * np.pi) * np.sin(sim.trajs[i] * np.pi)
    )
    H = [H0, [sds, lambda t, args: traj[int(t * sample_rate)] if t < t_max else 0], [H_control, drive_term]]
    result = qt.mesolve(H, initial_state, time_points, c_ops, options=opts)
    cavity_levels = sim.cavity_levels
    transmon_levels = sim.transmon_levels
    n_transmon = transmon_levels
    n_cavity = cavity_levels
    avg_values_single = np.zeros(len(time_points))
    for j, t in enumerate(time_points):
        floquet_states, sigmax, sigmas = sim.kick_and_sigmax(t)
        projectors = get_projector(floquet_states)
        P0k = projectors[cavity_idx-1]
        P1k = projectors[n_cavity + cavity_idx-1]
        P2k = projectors[2 * n_cavity + cavity_idx-1]
        avg_values_single[j] = (
            qt.expect(P0k, result.states[j])
            + qt.expect(P1k, result.states[j])
            + qt.expect(P2k, result.states[j])
        )
    return avg_values_single


def fit_T1_function(time_points, population_data):
    def fit1(t, T1):
        return 1 - np.exp(-t / T1)
    try:
        popt, _ = curve_fit(fit1, time_points, population_data, p0=[np.median(time_points)])
        return popt[0]
    except (RuntimeError, ValueError) as e:
        warnings.warn(f"T1 fit failed: {e}")
        return np.nan


def calculate_t1_uncertainty_via_bootstrap(t1_population_matrix, t1_time_points, fit_t1_func, num_bootstrap_samples=1000):
    num_t1_traj = t1_population_matrix.shape[0]
    t1_samples = []
    for _ in range(num_bootstrap_samples):
        idx = np.random.randint(0, num_t1_traj, size=num_t1_traj)
        avg_t1 = np.mean(t1_population_matrix[idx], axis=0)
        t1_val = fit_t1_func(t1_time_points, avg_t1)
        if np.isfinite(t1_val) and t1_val > 0:
            t1_samples.append(t1_val)
    if len(t1_samples) == 0:
        return np.nan, np.nan
    t1_samples = np.asarray(t1_samples)
    return np.mean(t1_samples), np.std(t1_samples, ddof=1)


def run_full_simulation(A, S0, initial_state_idx, cavity_idx):
    initial_state = []
    t_max = 100000
    num_realizations = 100
    sample_rate = 1
    sim = Simulation(A, initial_state, t_max, sample_rate, num_realizations, S0)
    sim.noise_check()

    t1_sim_time = 50
    num_trajectories = len(sim.trajs)
    t1_time_points = np.linspace(0, t1_sim_time, t1_sim_time)
    avg_values = Parallel(n_jobs=-1)(
        delayed(simulate_trajectory_T1)(i, sim, A, t_max, sample_rate, t1_sim_time, initial_state_idx, cavity_idx)
        for i in range(num_trajectories)
    )
    avg_values_T1 = np.array(avg_values)

    return avg_values_T1, t1_time_points


if __name__ == "__main__":
    total_start = time.perf_counter()

    A = 10e-3 * 2 * np.pi
    S0 = 1e-5

    print(f"Running T1 simulation: A/(2π) = {A/(2*np.pi)*1e3:.1f} mHz, S0 = {S0:.1e}")
    print(f"  levels=(transmon=3, cavity=5), t_max=100000, num_realizations=100, t1_sim_time=1000")

    for idx in [1,2,3,4]:
        print(f"\n{'='*50}")
        print(f"Initial Floquet state index = {idx}; fitting P0{idx}+P1{idx}+P2{idx}")
        avg_values_T1, t1_time_points = run_full_simulation(
            A, S0, initial_state_idx=idx, cavity_idx=idx
        )
        print(f"  avg_values_T1 shape: {avg_values_T1.shape}")
        print("  Running bootstrap (1000 samples) ...")
        T1_avg, T1_std = calculate_t1_uncertainty_via_bootstrap(
            avg_values_T1, t1_time_points, fit_T1_function
        )
        print(f"  T1(idx={idx}) = {T1_avg:.6e} ± {T1_std:.6e}")
        print(f"  1/T1(idx={idx}) = {1.0/T1_avg:.6e}")
        print(f"{'='*50}")

    print(f"Total runtime: {time.perf_counter() - total_start:.1f} s")
