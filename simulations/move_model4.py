import os
import pandas as pd
import numpy as np


def move_files_and_create_dataframe(source_folder, destination_folder, n0, n1, psi0, E0, model, target_N=2000):

    os.makedirs(destination_folder, exist_ok=True)

    subfolder = os.path.join(destination_folder, f"{int(n0)}_{int(n1)}")
    os.makedirs(subfolder, exist_ok=True)

    files = os.listdir(source_folder)

    ec_values = []
    sqrt_num_rows = []
    models = []
    psi_0_values = []
    psi_1_values = []
    E0_values = []
    n0_values = []
    n1_values = []
    dt_values = []

    processed_files = set()
    target_num_rows = target_N ** 2

    # ================================================================
    # LOOP OVER FILES
    # ================================================================

    for file_name in files:

        if model not in file_name:
            continue

        if f"psi0_{psi0}" not in file_name:
            continue

        if f"E0_{E0}" not in file_name:
            continue

        if f"n0_{n0}_n1_{n1}" not in file_name:
            continue

        if file_name in processed_files:
            continue

        source_path = os.path.join(source_folder, file_name)

        if not os.path.isfile(source_path):
            continue

        # ============================================================
        # CHECK N
        # ============================================================

        try:
            with open(source_path, "r") as file:
                num_rows = sum(1 for _ in file)

        except UnicodeDecodeError:
            print(f"UnicodeDecodeError occurred with file: {source_path}")
            continue

        if num_rows != target_num_rows:
            continue

        N = int(np.sqrt(num_rows))

        # ============================================================
        # HARD LINK FILE WITHOUT DUPLICATING DATA
        # ============================================================

        destination_path = os.path.join(subfolder, file_name)

        if os.path.exists(destination_path):

            if os.path.samefile(source_path, destination_path):
                print(f"Already linked, skipping: {file_name}")

            else:
                os.remove(destination_path)
                os.link(source_path, destination_path)
                print(f"Replaced existing copy with hard link: {file_name}")

        else:
            os.link(source_path, destination_path)
            print(f"Linked: {file_name}")

        print(f"    N = {N}")

        # ============================================================
        # EXTRACT PARAMETERS FROM FILENAME
        # ============================================================

        ec_value = float(file_name.split("_Ecouple_")[1].split("_")[0])

        if "_dt_" in file_name:
            dt = float(file_name.split("_dt_")[1].split("_")[0])
        else:
            dt = 0.001

        model_name = file_name.split("_")[0]
        file_psi0 = float(file_name.split("_psi0_")[1].split("_")[0])
        file_psi1 = float(file_name.split("_psi1_")[1].split("_")[0])

        print(f"{file_name[:6]}, n0={n0}, n1={n1}, N={N}, EC={ec_value}, dt={dt}")

        # ============================================================
        # STORE VALUES
        # ============================================================

        ec_values.append(ec_value)
        sqrt_num_rows.append(N)
        models.append(model_name)
        psi_0_values.append(file_psi0)
        psi_1_values.append(file_psi1)
        n0_values.append(n0)
        n1_values.append(n1)
        E0_values.append(E0)
        dt_values.append(dt)

        processed_files.add(file_name)

    # ================================================================
    # CREATE DATAFRAME
    # ================================================================

    df = pd.DataFrame({
        "n0": n0_values,
        "n1": n1_values,
        "Model": models,
        "Ec": ec_values,
        "SqrtNumRows": sqrt_num_rows,
        "Psi_0": psi_0_values,
        "Psi_1": psi_1_values,
        "E0": E0_values,
        "dt": dt_values
    })

    if df.empty:
        print(f"No N={target_N} files found for n0={n0}, n1={n1}, psi0={psi0}")
        return df

    # ================================================================
    # SAVE / UPDATE CSV
    # ================================================================

    csv_file_path = os.path.join(destination_folder, f"{int(n0)}_{int(n1)}.csv")

    if os.path.exists(csv_file_path):

        existing_df = pd.read_csv(csv_file_path)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df = combined_df.drop_duplicates()
        combined_df.to_csv(csv_file_path, index=False)

        print(f"Updated '{csv_file_path}'")

    else:

        df.to_csv(csv_file_path, index=False)
        print(f"Created '{csv_file_path}'")

    return df


# ================================================================
# MODEL 4 — LINK N = 2000 FILES
# ================================================================

target_dir = "/home/saraib/scratch/model4/n1_mixed"
target_dir_out = "/home/saraib/scratch/model4/moved_files_2000"

# ================================================================
# PARAMETERS
# ================================================================

n0_array = [float(n) for n in range(1, 31)]
n1_array = [3.0, 5.0, 7.0]

E0 = 2.0
model = "model4"
target_N = 2000

psi0_array = [8.0, 16.0, 24.0, 32.0, 64.0]

# ================================================================
# RUN
# ================================================================

for psi0 in psi0_array:

    print("\n")
    print("=" * 60)
    print(f"Processing psi0 = {psi0}")
    print("=" * 60)

    for n0 in n0_array:
        for n1 in n1_array:

            move_files_and_create_dataframe(
                source_folder=target_dir,
                destination_folder=target_dir_out,
                n0=n0,
                n1=n1,
                psi0=psi0,
                E0=E0,
                model=model,
                target_N=target_N
            )