import os
import shutil
import numpy as np

# Origin and output path
base_path = 'C:/Users/Diego/Desktop/N101/Datasets/Regression'  # Change this path to your origin folder
output_path = 'C:/Users/Diego/Desktop/N101/Datasets/Classification'  # Change this path to your output folder

# Obtain the list of STL folders
stl_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

# Create output folders if they do not exist
for axis in ['x', 'y', 'z']:
    os.makedirs(os.path.join(output_path, f'Img_axis_{axis}'), exist_ok=True)

# Initialize variables
num_stls = len(stl_folders)
samples_per_stl = 250
total_samples = num_stls * samples_per_stl
label_vector = np.zeros(total_samples, dtype=int)

# Global index for the vector rows
global_sample_index = 0

# Process every STL folder
for stl_index, stl_folder in enumerate(stl_folders):
    stl_path = os.path.join(base_path, stl_folder, "Img")
    
    # Process x, y, z axes
    for axis in ['x', 'y', 'z']:
        source_dir = os.path.join(stl_path, f'Img_axis_{axis}')
        dest_dir = os.path.join(output_path, f'Img_axis_{axis}')
        
        # Count pre-existing files in the output folder
        existing_files = len(os.listdir(dest_dir)) + 1
        
        # Obtain the first 250 files sorted by name
        source_files = sorted(os.listdir(source_dir))[:samples_per_stl]
        
        # Copy files
        for file_name in source_files:
            src_file = os.path.join(source_dir, file_name)
            
            # Generate the name for the output file
            dst_file = os.path.join(dest_dir, f"sample_{existing_files}_{axis}")
            base, ext = os.path.splitext(file_name)
            dst_file += ext  # Mantain the original extension
            
            # Copy file
            shutil.copy(src_file, dst_file)
            existing_files += 1

    # Update labels vector
    start_index = global_sample_index
    end_index = global_sample_index + samples_per_stl
    label_vector[start_index:end_index] = stl_index
    global_sample_index += samples_per_stl

# Save the vector as a .csv file
output_csv = os.path.join(output_path, "labels_vector.csv")
np.savetxt(output_csv, label_vector, delimiter=",", fmt="%d")

# Save the vector as a .npy file
output_npy = os.path.join(output_path, "labels_vector.npy")
np.save(output_npy, label_vector)

print(f"Vector de etiquetas guardado en {output_csv} y {output_npy}")
print("Consolidación completada.")
