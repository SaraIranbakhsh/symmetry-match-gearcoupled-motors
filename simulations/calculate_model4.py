import pyximport
pyximport.install()

from utilities import calc_flux
import os
import pandas as pd
import numpy as np
from numpy import empty
from math import pi
from numpy import (
    array, linspace, arange, loadtxt, vectorize, pi as npi, exp, zeros,
    empty, log, log2, finfo, true_divide, asarray, where, partition, isnan, nan,
    ones, argmax, argmin, set_printoptions, diagonal, round, inf, concatenate, sin, cos
    )

from scipy.integrate import trapezoid as trapz


def set_params(model, n0, n1, f0, E0=2.0, E1=2.0):
    N = 100

    Ecouple = 20.0

    psi_0 = f0
    psi_1 = -2.0
    psi_0_n = f0
    psi_1_n = -2.0

    # model4: scaling driving — force scales with subunit number
    if model == "model4" or model == "model2":
        psi_0_n = n0 * psi_0 / 8
        psi_1_n = n1 * psi_1 / 3

    m0 = 1.0
    m1 = 1.0
    beta = 1.0
    gamma = 1000.0
    phase = 0.0

    return (
        E0, Ecouple, E1, psi_0, psi_1, psi_0_n, psi_1_n,
        n0, n1,
        m0, m1, phase, beta, gamma, model
        )


def calculate_flux_power_and_efficiency(
    target_dir,
    target_dir_out,
    Ecouple_array,
    params,
    N_array,
    dt=0.001
):

    [
        E0, __, E1, psi_0, psi_1, psi_0_n, psi_1_n, n0, n1,
        m0, m1, phase, beta, gamma, model
    ] = params

    input_file_name = (
        "/" + model
        + "_t_reference_E0_{0}_Ecouple_{1}_E1_{2}_psi0_{3}_psi1_{4}_n0_{5}_n1_{6}_phase_{7}_outfile.dat"
    )

    output_file_name = (
        "/" + model + "_flux_power_efficiency_"
        + "E0_{0}_E1_{1}_psi0_{2}_psi1_{3}_n0_{4}_n1_{5}_phase_{6}_outfile.dat"
    )

    efficiency_ratio_mod = []

    l = len(Ecouple_array)

    integrate_flux_X = empty(l)
    integrate_flux_Y = empty(l)
    integrate_power_X = empty(l)
    integrate_power_Y = empty(l)
    efficiency_ratio = empty(l)

    for ii, Ec in enumerate(Ecouple_array):

        N = int(N_array[ii])
        dx = 2 * pi / N
        positions = linspace(0.0, 2 * pi - dx, N)

        flux_array = empty((2, N, N))

        try:
            data_array = loadtxt(
                target_dir + input_file_name.format(
                    E0,
                    Ec,
                    E1,
                    psi_0,
                    psi_1,
                    n0,
                    n1,
                    phase
                ),
                usecols=(0, 3, 4, 5, 6, 7, 8)
            )

        except:
            continue

        prob_ss_array = data_array[:, 0].reshape((N, N))

        print(
            "normalized?",
            prob_ss_array.sum(axis=None)
        )

        drift_at_pos = (
            data_array[:, 1:3]
            .T
            .reshape((2, N, N))
        )

        diffusion_at_pos = (
            data_array[:, 3:]
            .T
            .reshape((4, N, N))
        )

        calc_flux(
            positions,
            prob_ss_array,
            drift_at_pos,
            diffusion_at_pos,
            flux_array,
            N,
            dx
        )

        flux_array = asarray(flux_array) / (dx * dx)

        integrate_flux_X[ii] = (
            (1. / (2 * pi))
            * trapz(
                trapz(
                    flux_array[0, ...],
                    dx=dx,
                    axis=1
                ),
                dx=dx
            )
        )

        integrate_flux_Y[ii] = (
            (1. / (2 * pi))
            * trapz(
                trapz(
                    flux_array[1, ...],
                    dx=dx,
                    axis=0
                ),
                dx=dx
            )
        )

        integrate_power_X[ii] = (
            integrate_flux_X[ii] * psi_0_n
        )

        integrate_power_Y[ii] = (
            integrate_flux_Y[ii] * psi_1_n
        )

        if (
            abs(integrate_power_X[ii])
            <= abs(integrate_power_Y[ii])
        ):
            efficiency_ratio_mod.append(
                -(
                    integrate_power_X[ii]
                    / integrate_power_Y[ii]
                )
            )
        else:
            efficiency_ratio_mod.append(
                -(
                    integrate_power_Y[ii]
                    / integrate_power_X[ii]
                )
            )

    efficiency_ratio = np.zeros(
        len(integrate_power_Y)
    )

    if abs(psi_0_n) <= abs(psi_1_n):

        efficiency_ratio = -(
            integrate_power_X
            / integrate_power_Y
        )

    else:

        efficiency_ratio = -(
            integrate_power_Y
            / integrate_power_X
        )

    print(
        model
        + " n0 = {0} n1= {1} f0= {2:.2f}, <j0>={3}".format(
            n0,
            n1,
            psi_0_n,
            [
                format(value, ".14e")
                for value in integrate_flux_X
            ]
        )
    )

    print(
        model
        + " n0 = {0} n1= {1} f1= {2:.2f}, <j1>={3}".format(
            n0,
            n1,
            psi_1_n,
            [
                format(value, ".14e")
                for value in integrate_flux_Y
            ]
        )
    )

    print(
        "-------------------------------------------------------------------------------------"
    )

    with open(
        target_dir_out
        + output_file_name.format(
            E0,
            E1,
            psi_0,
            psi_1,
            n0,
            n1,
            phase
        ),
        "w"
    ) as ofile:

        for ii, Ec in enumerate(Ecouple_array):

            ofile.write(
                f"{Ec:.15e}" + "\t"
                + f"{integrate_flux_X[ii]:.15e}" + "\t"
                + f"{integrate_flux_Y[ii]:.15e}" + "\t"
                + f"{integrate_power_X[ii]:.15e}" + "\t"
                + f"{integrate_power_Y[ii]:.15e}" + "\t"
                + f"{efficiency_ratio[ii]:.15e}" + "\t"
                + f"{efficiency_ratio_mod[ii]:.15e}" + "\n"
            )

        print(
            "saved",
            ofile
        )

        ofile.flush()

    ofile.close()

    with open(
        target_dir_out
        + output_file_name.format(
            E0,
            E1,
            psi_0,
            psi_1,
            n0,
            n1,
            phase
        ),
        "r"
    ) as ofile:

        first_line = ofile.readline()
        print(first_line)


# ── parameters ───────────────────────────────────────────────────────────────

model_array = ["model4"]

target_dir = (
    "/home/saraib/scratch/model4/moved_files_2000/"
)

target_dir_out = (
    "/home/saraib/scratch/model4/pfe_2000/"
)


# Full list of psi_0 values
psi_0_array = [
    8.0,
    16.0,
]


E0_fixed = 2.0
dt = 0.001


n0_array = [
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    11.0,
    12.0,
    13.0,
    14.0,
    15.0,
    16.0,
    17.0,
    18.0,
    19.0,
    20.0,
    21.0,
    22.0,
    23.0,
    24.0,
    25.0,
    26.0,
    27.0,
    28.0,
    29.0,
    30.0
]


n1_array = [
    3.0,
    5.0,
    7.0
]


# ── main loop ────────────────────────────────────────────────────────────────

for psi_0 in psi_0_array:

    for n1 in n1_array:

        for n0 in n0_array:

            n0n1_str = (
                str(int(n0))
                + "_"
                + str(int(n1))
            )

            for model in model_array:

                df_file_path = (
                    target_dir
                    + f'{int(n0)}_{int(n1)}.csv'
                )

                print(df_file_path)

                if not os.path.exists(df_file_path):

                    print(
                        f"DataFrame file for "
                        f"n0={n0}, n1={n1} "
                        f"does not exist. Skipping..."
                    )

                    continue

                df = pd.read_csv(
                    df_file_path
                )

                filtered_df = df[
                    (df['n0'] == n0)
                    & (df['n1'] == n1)
                    & (df['Model'] == model)
                    & (df['Psi_0'] == psi_0)
                    & (df['E0'] == E0_fixed)
                ]

                n_before = len(
                    filtered_df
                )

                filtered_df = (
                    filtered_df.drop_duplicates(
                        subset=[
                            'Ec',
                            'SqrtNumRows',
                            'E0',
                            'Psi_0',
                            'n0',
                            'n1'
                        ]
                    )
                )

                n_after = len(
                    filtered_df
                )

                if n_before > n_after:

                    print(
                        f"Warning: dropped "
                        f"{n_before - n_after} "
                        f"duplicate entries for "
                        f"n0={n0}, n1={n1}"
                    )

                Ecouple_array = (
                    filtered_df['Ec'].values
                )

                N_array = (
                    filtered_df['SqrtNumRows'].values
                )

                E0_array = (
                    filtered_df['E0'].values
                )

                sorted_indices = np.argsort(
                    Ecouple_array
                )

                Ec_array_sorted = (
                    Ecouple_array[
                        sorted_indices
                    ]
                )

                N_array_sorted = (
                    N_array[
                        sorted_indices
                    ]
                )

                print(
                    Ec_array_sorted
                )

                print(
                    N_array_sorted
                )

                params = set_params(
                    model,
                    n0,
                    n1,
                    psi_0,
                    E0_fixed,
                    E0_fixed
                )

                if not os.path.exists(
                    target_dir_out
                    + n0n1_str
                ):

                    os.makedirs(
                        target_dir_out
                        + n0n1_str
                    )

                calculate_flux_power_and_efficiency(
                    target_dir
                    + n0n1_str,

                    target_dir_out
                    + n0n1_str,

                    Ec_array_sorted,

                    params,

                    N_array_sorted,

                    dt
                )