#import !/usr/bin/env/python3
import os
from math import pi
from numpy import finfo, asarray, empty, zeros, linspace
from datetime import datetime

import pyximport
pyximport.install()

import fpe_model4
from fpe_model4 import launchpad_reference




dt = 0.001  # time discretization. Keep this number low
N = 2000  # inverse space discretization. Keep this number high!

# model constants
beta = 1.0  # thermodynamic beta: 1/kT
m0 = m1 = 1.0  # masses of Ro and R1

# model-specific parameters
gamma0 = gamma1 = 1000.0  # drag coefficients of Ro and R1

# ---- parameters modified by parallelize.sh (lines 26-34) ----
E0 = 2.0  # energy scale of Ro
Ecouple = 2.0
E1 = 2.0  # energy scale of R1
mu_Hp = 4.0  # mu_{H+}: driving force on Ro
mu_atp = -2.0  # mu_{ATP}: driving force on R1

n0 = 3.0  # number of minima in the potential of Ro
n1 = 3.0  # number of minima in the potential of R1
phase = 0.0  # offset between sub-systems
# --------------------------------------------------------------

dx = (2 * pi) / N  # space discretization: total distance / number of points

# CFL stability check
if E0 == 0.0 and E1 == 0.0:
    time_check = 100000000.0
else:
    time_check = dx / (
        abs(0.5 * (n0 * Ecouple + E0 * n0) - (n0 * mu_Hp / 8.0)) / (m0 * gamma0)
        + abs(0.5 * (-n1 * Ecouple + E1 * n1) - (n1 * mu_atp / 3.0)) / (m1 * gamma1)
    )
    print(time_check)

if dt > time_check:
    print(time_check)
    print("!!!TIME UNSTABLE!!! No use in going on. Aborting...\n")
    exit(1)

check_step = int(2.0 / dt)
print(f"Number of times before check = {check_step}")

prob = zeros((N, N))
p_now = zeros((N, N))
p_last = zeros((N, N))
p_last_ref = zeros((N, N))
positions = linspace(0, (2 * pi) - dx, N)
potential_at_pos = zeros((N, N))
drift_at_pos = zeros((2, N, N))
diffusion_at_pos = zeros((4, N, N))

print(
    f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
    + "Launching FPE simulation..."
)
launchpad_reference(
    n0, n1,
    phase,
    positions,
    prob, p_now,
    p_last, p_last_ref,
    potential_at_pos,
    drift_at_pos,
    diffusion_at_pos,
    N, dx, check_step,
    E0, Ecouple, E1, mu_Hp, mu_atp,
    dt, m0, m1, beta, gamma0, gamma1
)
print(
    f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
    + "FPE simulation done!"
)

print(
    f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
    + "Processing data..."
)

p_now = asarray(p_now)
p_equil = asarray(prob)
potential_at_pos = asarray(potential_at_pos)
drift_at_pos = asarray(drift_at_pos)
diffusion_at_pos = asarray(diffusion_at_pos)

assert (p_now >= 0.0).all(), \
    "ABORT: Probability density has negative values!"
assert (abs(p_now.sum(axis=None) - 1.0).__abs__() <= finfo('float32').eps), \
    "ABORT: Probability density is not normalized!"

print(
    f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
    + "Processing finished!"
)

print(
    f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Saving data..."
)

data_dir = '/home/saraib/scratch/model4/n1_mixed/'

data_filename = (
    f"/model4_t_reference_E0_{E0}_Ecouple_{Ecouple}_E1_{E1}_"
    + f"psi0_{mu_Hp}_psi1_{mu_atp}_"
    + f"n0_{n0}_n1_{n1}_phase_{phase}_"
    + "outfile.dat")
data_total_path = data_dir + data_filename

with open(data_total_path, 'w') as dfile:
    for i in range(N):
        for j in range(N):
            dfile.write(
                f'{p_now[i, j]:.15e}'
                + '\t' + f'{p_equil[i, j]:.15e}'
                + '\t' + f'{potential_at_pos[i, j]:.15e}'
                + '\t' + f'{drift_at_pos[0, i, j]:.15e}'
                + '\t' + f'{drift_at_pos[1, i, j]:.15e}'
                + '\t' + f'{diffusion_at_pos[0, i, j]:.15e}'
                + '\t' + f'{diffusion_at_pos[1, i, j]:.15e}'
                + '\t' + f'{diffusion_at_pos[2, i, j]:.15e}'
                + '\t' + f'{diffusion_at_pos[3, i, j]:.15e}'
                + '\n'
            )

print(
    f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Saving completed!"
)
print("Exiting...")
