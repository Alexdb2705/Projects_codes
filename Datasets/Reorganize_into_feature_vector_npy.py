import os
import shutil
import numpy as np
import glob
from random import shuffle

def reorganization(samples_per_stl, nf, nd, datasets_path):
    # Origin and output path
    base_path = os.path.join(datasets_path, f"Raw/Samples_{nf}_f_{nd}_d")  # Path to folder where separated datasets are stored
    output_path = os.path.join(datasets_path, f"Reorganized/Classification_{samples_per_stl}_{nf}_f_{nd}_d")  # Path to folder where reorganized dataset will be saved
    labels_path = os.path.join(datasets_path, "Reorganized/Labels_vector")  # Path to folder where labels_vectors are stored

    # Obtain the list of STL folders
    stl_folders = []
    for root, dirs, files in os.walk(base_path):
        if any(file.endswith(".npy") for file in files):
            stl_folders.append(root.split('/')[-1])
    print("stls",stl_folders)
    # If a folder with the exact same name or path exists, deletes it and creates a new empty folder with that name
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    existing_files = len(glob.glob(os.path.join(output_path, "sample_*")))
    num_stls = len(stl_folders)
    # existing_files_per_stl = int((existing_files) / num_stls)

    # Initialize variables
    total_samples = num_stls * samples_per_stl
    # label_vector = np.zeros(total_samples + existing_files, dtype=int)
    label_vector = np.zeros(total_samples, dtype=int)
    # if existing_files >= 1:
    #     label_vector_old = np.load(os.path.join(labels_path,f"labels_vector_{samples_per_stl}_{nf}_f_{nd}_d.npy"))
    #     label_vector[:existing_files] = label_vector_old

    # Process every STL folder
    for stl_index, stl_folder in enumerate(stl_folders):
        stl_path = os.path.join(base_path, stl_folder)
        
        # Obtain the first samples_per_stl files sorted by name
        source_files = sorted(glob.glob(os.path.join(stl_path, "*.npy")))[:samples_per_stl]

        start_index = existing_files
        end_index = samples_per_stl + existing_files
        label_vector[start_index:end_index] = stl_index
        
        # Copy files
        for file in source_files:
            
            # Generate the name for the output file
            name_parts = file.split('_')
            dst_file = os.path.join(output_path, f"sample_{existing_files+1}_{name_parts[-2]}_{name_parts[-1]}")

            # Copy file
            shutil.copy(file, dst_file)
            existing_files += 1

    # Save the vector as a .csv file
    output_csv = os.path.join(labels_path, f"labels_vector_{samples_per_stl}_{nf}_f_{nd}_d.csv")
    np.savetxt(output_csv, label_vector, delimiter=",", fmt="%d")

    # Save the vector as a .npy file
    output_npy = os.path.join(labels_path, f"labels_vector_{samples_per_stl}_{nf}_f_{nd}_d.npy")
    np.save(output_npy, label_vector)

    print(f"Vector de etiquetas guardado en {output_csv} y {output_npy}")
    print("Consolidación completada.")
