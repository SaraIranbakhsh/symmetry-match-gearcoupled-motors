#import !/usr/bin/env/python3
import os
from math import pi
from numpy import finfo, asarray, empty, zeros, linspace
from datetime import datetime


import fpe_model3
from fpe_model3 import launchpad_reference









dt = 0.001  # time discretization. Keep this number low
N = 2000  # inverse space discretization. Keep this number high!

# model constants
beta = 1.0  # thermodynamic beta: 1/kT
m0 = m1 = 1.0  # masses of Fo and F1

# model-specific parameters
gamma0 = gamma1= 1000.0  # drag coefficients of Fo and F1

E0 = 2.0 # energy scale of Fo
Ecouple = 2.0
E1 = 2.0 # energy scale of F1
mu_Hp = 4.0 #  mu_{H+}: energy INTO (positive) Fo by F1
mu_atp = -2.0 # mu_{ATP}: energy INTO (positive) F1 by Fo

n0 = 3.0  # number of minima in the potential of Fo
n1 = 3.0  # number of minima in the potential of F1
phase = 0.0  # how much sub-systems are offset from one another


dx = (2*pi) / N  # space discretization: total distance / number of points

# provide CFL criteria to make sure simulation doesn't blow up
if E0 == 0.0 and E1 == 0.0:
    time_check = 100000000.0
else:
    time_check = dx/(
        abs(0.5*(n0*Ecouple+E0*n0)-mu_Hp)/(m0*gamma0)
        + abs(0.5*(-n1*Ecouple+E1*n1)-mu_atp)/(m1*gamma1)
    )
    print(time_check)

if dt > time_check:
    print(time_check)
    # bail if user is stupid
    print("!!!TIME UNSTABLE!!! No use in going on. Aborting...\n")
    exit(1)

# how many time update steps before checking for steady state convergence
# enforce steady state convergence check every unit time
check_step = int(2.0/dt)

print(f"Number of times before check = {check_step}")

prob = zeros((N, N))
p_now = zeros((N, N))
p_last = zeros((N, N))
p_last_ref = zeros((N, N))
positions = linspace(0, (2*pi)-dx, N)
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
# recast everything into a numpy array
p_now = asarray(p_now)
p_equil = asarray(prob)
potential_at_pos = asarray(potential_at_pos)
drift_at_pos = asarray(drift_at_pos)
diffusion_at_pos = asarray(diffusion_at_pos)

# checks to make sure nothing went weird: bail at first sign of trouble
# check the non-negativity of the distribution
assert (p_now >= 0.0).all(), \
    "ABORT: Probability density has negative values!"
# check the normalization
assert (abs(p_now.sum(axis=None) - 1.0).__abs__() <= finfo('float32').eps), \
    "ABORT: Probability density is not normalized!"

print(
    f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
    + "Processing finished!"
)

# write to file
print(
    f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Saving data..."
)

# diffusion_at_pos slots after fpe_model3 edit:
# [0] = D11, [1] = D22, [2] and [3] unused (zeros)
# Write in original 4-column order [D11, 0, 0, D22] so that the analysis
# pipeline (calc_flux in utilities.pyx) receives the expected slot layout.
data_dir = '/home/saraib/scratch/model3/n1_mixed/'
data_filename = (
    f"/model3_t_reference_E0_{E0}_Ecouple_{Ecouple}_E1_{E1}_"
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
                + '\t' + f'{diffusion_at_pos[0, i, j]:.15e}'  # D11
                + '\t' + f'{0.0:.15e}'                         # D12 = 0
                + '\t' + f'{0.0:.15e}'                         # D21 = 0
                + '\t' + f'{diffusion_at_pos[1, i, j]:.15e}'  # D22
                + '\n'
            )

print(
    f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Saving completed!"
)

print("Exiting...")