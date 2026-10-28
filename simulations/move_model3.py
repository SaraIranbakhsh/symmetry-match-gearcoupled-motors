import os
import shutil
import pandas as pd
import numpy as np



def move_files_and_create_dataframe(source_folder, destination_folder, n0, n1,psi0,E0,model):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    # Create the subfolder based on n0 and n1
    subfolder = os.path.join(destination_folder, f"{int(n0)}_{int(n1)}")
    os.makedirs(subfolder, exist_ok=True)
    
    # Get the list of files in the source folder
    files = os.listdir(source_folder)
    
    # Initialize lists to store the Ec values, square root of the number of rows, model, psi_0, and psi_1
    ec_values = []
    sqrt_num_rows = []
    models = []
    psi_0_values = []
    psi_1_values = []
    E0_values = []
    n0_values = []
    n1_values = []
    dt_values = []
    
    # Set to keep track of processed file names
    processed_files = set()
    
    # Iterate over the files
    for file_name in files:
        if model in file_name:
            if f"psi0_{psi0}" in file_name and f"E0_{E0}" in file_name:
        # Check if the file matches the pattern and has not been processed before
                if f"n0_{n0}_n1_{n1}" in file_name and file_name not in processed_files:
                    # Move the file to the subfolder
                    source_path = os.path.join(source_folder, file_name)
                    print(source_path)
                    destination_path = os.path.join(subfolder, file_name)
                    shutil.copy(source_path, destination_path)
                    #print(f"File '{file_name}' copied to '{subfolder}'")
                    
                    # Count the number of rows in the file
                    # with open(destination_path, 'r') as file:
                    #     num_rows = sum(1 for line in file)
                    #     sqrt_num_rows.append(np.sqrt(num_rows))
                    #     #print(f"N: {sqrt_num_rows[-1]}")
                    try:
                        with open(destination_path, 'r') as file:
                            num_rows = sum(1 for line in file)
                            sqrt_num_rows.append(np.sqrt(num_rows))
                    except UnicodeDecodeError:
                        print(f"UnicodeDecodeError occurred with file: {destination_path}")
                        continue  # Move to the next file
                    
                    # Extract the Ec value from the file name
                    ec_value = float(file_name.split("_Ecouple_")[1].split("_")[0])
                    if "dt" in file_name:
                        dt = float(file_name.split("_dt_")[1].split("_")[0])
                        print(fr"{file_name[:6]}, n0={n0},n1={n1},N={np.sqrt(num_rows)},EC={ec_value},dt={dt}")
                    else:
                        print(fr"{file_name[:6]}, n0={n0},n1={n1},N={np.sqrt(num_rows)},EC={ec_value}")
                    
                    # Extract the model, psi_0, and psi_1 from the file name
                    model = file_name.split("_")[0]
                    psi_0 = float(file_name.split("_psi0_")[1].split("_")[0])
                    psi_1 = float(file_name.split("_psi1_")[1].split("_")[0])
                    
                    # Append the values to the respective lists
                    ec_values.append(ec_value)
                    models.append(model)
                    psi_0_values.append(psi_0)
                    psi_1_values.append(psi_1)
                    n0_values.append(n0)
                    E0_values.append(E0)
                    n1_values.append(n1)
                    if "dt" in file_name:
                        dt = float(file_name.split("_dt_")[1].split("_")[0])
                        print(fr"{file_name[:6]}, n0={n0},n1={n1},N={np.sqrt(num_rows)},EC={ec_value},dt={dt}")
                        dt_values.append(dt)
                    else:
                        dt_values.append(0.001)
                        print(fr"{file_name[:6]}, n0={n0},n1={n1},N={np.sqrt(num_rows)},EC={ec_value}")
                    
                    # Add the file name to the processed files set
                    processed_files.add(file_name)
    # Create a pandas DataFrame with the Ec values, square root of the number of rows, model, psi_0, and psi_1
    df = pd.DataFrame({"n0": n0_values, "n1": n1_values, 'Model': models, 'Ec': ec_values, 'SqrtNumRows': sqrt_num_rows, 'Psi_0': psi_0_values, 'Psi_1': psi_1_values, 'E0': E0_values,'dt': dt_values})
    # Check if the CSV file already exists
    csv_file_path = os.path.join(destination_folder, f"{int(n0)}_{int(n1)}.csv")
    if os.path.exists(csv_file_path):
        # Read the existing CSV file
        existing_df = pd.read_csv(csv_file_path)
        
        # Check if the contents of the CSV file are the same as the data to be added
        if existing_df.equals(df):
            print(f"Skipping file '{file_name}' as it already exists with the same data")
        else:
            # Append the new data to the existing CSV file
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(csv_file_path, index=False)
            print(f"Data appended to '{csv_file_path}'")
    else:
        df.to_csv(os.path.join(destination_folder, f'{int(n0)}_{int(n1)}.csv'), index=False)

    
    return df



####Copies files from downloaded folder to specific direction#####
n0_array = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0]


#target_dir = "/project/6007000/saraib/src/working_directory_cython/model3/new_data_n1_5_1000_fo_4_8"
#target_dir_out = "/project/6007000/saraib/src/working_directory_cython/model3/new_data_n1_5_1000_fo_4_8/moved_files"

####################
target_dir = "/project/6007000/saraib/src/working_directory_cython/model3/new_data_1000_fo_8"
target_dir_out = "/project/6007000/saraib/src/working_directory_cython/model3/new_data_1000_fo_8/moved_files"
####################
target_dir = "/project/6007000/saraib/src/working_directory_cython/model3/new_data_n1_3_1000_fo_5_6_7"
target_dir_out = "/project/6007000/saraib/src/working_directory_cython/model3/new_data_n1_3_1000_fo_5_6_7/moved_files"
##################
target_dir = "/project/6007000/saraib/src/working_directory_cython/model3/new_data_1000_fo_8"
target_dir_out = "/project/6007000/saraib/src/working_directory_cython/model3/new_data_1000_fo_8/moved_files"




target_dir = "/home/saraib/scratch/model3/n1_mixed"
target_dir_out = "/home/saraib/scratch/model3/moved_files_2000"


n1_array = [3.0,5.0,7.0]
n0_array=[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0]

#n0_array=[1.0]

    
        
psi0= 8.0
E0= 2.0

for n0 in n0_array:
    for n1 in n1_array:
        move_files_and_create_dataframe(target_dir, target_dir_out, n0, n1, psi0, E0,"model3")
        
        
        
psi0= 24.0
E0= 2.0

for n0 in n0_array:
    for n1 in n1_array:
        move_files_and_create_dataframe(target_dir, target_dir_out, n0, n1, psi0, E0,"model3")
        
        
        
        
        
psi0= 16.0
E0= 2.0

for n0 in n0_array:
    for n1 in n1_array:
        move_files_and_create_dataframe(target_dir, target_dir_out, n0, n1, psi0, E0,"model3")
        
        
        
psi0= 32.0
E0= 2.0

for n0 in n0_array:
    for n1 in n1_array:
        move_files_and_create_dataframe(target_dir, target_dir_out, n0, n1, psi0, E0,"model3")
        
        
        
        
        
        
        
psi0= 64.0
E0= 2.0

for n0 in n0_array:
    for n1 in n1_array:
        move_files_and_create_dataframe(target_dir, target_dir_out, n0, n1, psi0, E0,"model3")


