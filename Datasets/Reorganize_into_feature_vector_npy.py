import os
import shutil
import numpy as np
import glob
from random import shuffle
import json

def reorganization(scan_angle, nr_dictionary, nf, nd, datasets_path, top_folder_path, cw, snr):
    # Origin and output path

    total_samples = 0
    for i in nr_dictionary.values():
        total_samples += (i[1] - i[0])
        
    for c in range(50):

        if cw == 0:
            if snr == None:
                output_path = os.path.join(datasets_path, f"Reorganized/Classification_{total_samples}_{c}_{nf}_f_{nd}_d")
                output_csv = os.path.join(output_path, f"labels_vector_{total_samples}_{c}_{nf}_f_{nd}_d.csv")
                output_npy = os.path.join(output_path, f"labels_vector_{total_samples}_{c}_{nf}_f_{nd}_d.npy")
            else:
                output_path = os.path.join(datasets_path, f"Reorganized/Classification_{total_samples}_{c}_{nf}_f_{nd}_d_SNR_{snr}")
                output_csv = os.path.join(output_path, f"labels_vector_{total_samples}_{c}_{nf}_f_{nd}_d_SNR_{snr}.csv")
                output_npy = os.path.join(output_path, f"labels_vector_{total_samples}_{c}_{nf}_f_{nd}_d_SNR_{snr}.npy")
        else:
            if snr == None:
                output_path = os.path.join(datasets_path, f"Reorganized/Classification_{total_samples}_{c}_{nf}_f_{nd}_d_POV_{cw}")
                output_csv = os.path.join(output_path, f"labels_vector_{total_samples}_{c}_{nf}_f_{nd}_d_POV_{cw}.csv")
                output_npy = os.path.join(output_path, f"labels_vector_{total_samples}_{c}_{nf}_f_{nd}_d_POV_{cw}.npy")
            else:
                output_path = os.path.join(datasets_path, f"Reorganized/Classification_{total_samples}_{c}_{nf}_f_{nd}_d_POV_{cw}_SNR_{snr}")
                output_csv = os.path.join(output_path, f"labels_vector_{total_samples}_{c}_{nf}_f_{nd}_d_POV_{cw}_SNR_{snr}.csv")
                output_npy = os.path.join(output_path, f"labels_vector_{total_samples}_{c}_{nf}_f_{nd}_d_POV_{cw}_SNR_{snr}.npy")

        if os.path.isdir(output_path):
            pass
        else:
            os.makedirs(output_path, exist_ok=True)
            break

    assert c<50 , "Error in folder creation"

    with open(os.path.join(output_path, f"dictionary_{total_samples}_{c}_{nf}_f_{nd}_d.json"), 'w') as json_file:
        json.dump(nr_dictionary, json_file)

    # Obtain the list of STL folders
    stl_folders = []
    for g_name in nr_dictionary.keys():
        stl_folders.append(g_name)

    existing_files = len(glob.glob(os.path.join(output_path, "sample_*")))

    # Initialize variables
    label_vector = np.zeros(total_samples, dtype=int)

    def files_sorted(path, ending):
        names = glob.glob(os.path.join(path, f"*{ending}"))
        names.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
        return names

    # Process every STL folder
    for stl_index, stl_name in enumerate(stl_folders):
        stl_path = os.path.join(top_folder_path, stl_name)
        
        # Obtain the first samples_per_stl files sorted by name
        source_files_isar = files_sorted(stl_path, "fft.npy")[nr_dictionary[stl_name][0]:nr_dictionary[stl_name][1]]
        source_files_npy = files_sorted(stl_path, f"{scan_angle[0].upper()}.npy")[nr_dictionary[stl_name][0]:nr_dictionary[stl_name][1]]
        source_files_png = files_sorted(stl_path, f"{scan_angle[0].upper()}.png")[nr_dictionary[stl_name][0]:nr_dictionary[stl_name][1]]
        source_files_perfil = files_sorted(stl_path, "perfil.npy")[nr_dictionary[stl_name][0]:nr_dictionary[stl_name][1]]
        source_files_perfil_png = files_sorted(stl_path, "perfil.png")[nr_dictionary[stl_name][0]:nr_dictionary[stl_name][1]]
        source_files_field_npy = files_sorted(stl_path, "field.npy")[nr_dictionary[stl_name][0]:nr_dictionary[stl_name][1]]
        source_files_field_amp_png = files_sorted(stl_path, "amp.png")[nr_dictionary[stl_name][0]:nr_dictionary[stl_name][1]]
        source_files_field_ph_png = files_sorted(stl_path, "ph.png")[nr_dictionary[stl_name][0]:nr_dictionary[stl_name][1]]
        source_files_field_amp_npy = files_sorted(stl_path, "amp.npy")[nr_dictionary[stl_name][0]:nr_dictionary[stl_name][1]]
        source_files_field_ph_npy = files_sorted(stl_path, "ph.npy")[nr_dictionary[stl_name][0]:nr_dictionary[stl_name][1]]

        start_index = existing_files
        end_index = nr_dictionary[stl_name][1] - nr_dictionary[stl_name][0] + existing_files
        label_vector[start_index:end_index] = stl_index

        # Copy files
        for file in range(len(source_files_npy)):
            
            # Generate the name for the output file
            dst_file_npy = os.path.join(output_path, f"sample_{existing_files+1}.npy")
            dst_file_isar = os.path.join(output_path, f"sample_{existing_files+1}_fft.npy")
            dst_file_png = os.path.join(output_path, f"sample_{existing_files+1}.png")
            dst_file_perfil = os.path.join(output_path, f"sample_{existing_files+1}_fft_perfil.npy")
            dst_file_perfil_png = os.path.join(output_path, f"sample_{existing_files+1}_perfil.png")
            dst_file_field_npy = os.path.join(output_path, f"sample_{existing_files+1}_field.npy")
            dst_file_field_amp_png = os.path.join(output_path, f"sample_{existing_files+1}_field_amp.png")
            dst_file_field_ph_png = os.path.join(output_path, f"sample_{existing_files+1}_field_ph.png")
            dst_file_field_amp_npy = os.path.join(output_path, f"sample_{existing_files+1}_field_amp.npy")
            dst_file_field_ph_npy = os.path.join(output_path, f"sample_{existing_files+1}_field_ph.npy")

            # Copy file
            shutil.copy(source_files_npy[file], dst_file_npy)
            shutil.copy(source_files_isar[file], dst_file_isar)
            shutil.copy(source_files_png[file], dst_file_png)
            shutil.copy(source_files_perfil[file], dst_file_perfil)
            shutil.copy(source_files_perfil_png[file], dst_file_perfil_png)
            shutil.copy(source_files_field_npy[file], dst_file_field_npy)
            shutil.copy(source_files_field_amp_png[file], dst_file_field_amp_png)
            shutil.copy(source_files_field_ph_png[file], dst_file_field_ph_png)
            shutil.copy(source_files_field_amp_npy[file], dst_file_field_amp_npy)
            shutil.copy(source_files_field_ph_npy[file], dst_file_field_ph_npy)
            existing_files += 1

    # Save the vector as a .csv file
    # output_csv = os.path.join(output_path, f"labels_vector_{total_samples}_{c}_{nf}_f_{nd}_d.csv")
    np.savetxt(output_csv, label_vector, delimiter=",", fmt="%d")

    # Save the vector as a .npy file
    # output_npy = os.path.join(output_path, f"labels_vector_{total_samples}_{c}_{nf}_f_{nd}_d.npy")
    np.save(output_npy, label_vector)

    print(f"Vector de etiquetas guardado en {output_csv} y {output_npy}")
    print("\nConsolidación completada.")

def multiple_reorganization(nr_list, g_list):
    assert len(nr_list) == 2*len(g_list) or len(nr_list) == len(g_list) or len(nr_list) == 1 or len(nr_list) == 2, "Error in multiple reorganization"
    nr_dict = {}

    if len(nr_list) == 2*len(g_list):
        for i in range(len(g_list)):
            nr_dict[g_list[i]] = (nr_list[2*i], nr_list[2*i+1])
        return nr_dict
    elif len(nr_list) == len(g_list):
        for i in range(len(g_list)):
            nr_dict[g_list[i]] = (0, nr_list[i])
        return nr_dict
    elif len(nr_list) == 1:
        for i in range(len(g_list)):
            nr_dict[g_list[i]] = (0, nr_list[0])
        return nr_dict
    elif len(nr_list) == 2:
        for i in range(len(g_list)):
            nr_dict[g_list[i]] = (nr_list[0], nr_list[1])
        return nr_dict

