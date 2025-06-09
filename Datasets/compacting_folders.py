import os
import glob
import shutil

samples_folder = "/home/newfasant/N101-IA/Datasets/Raw/Samples_64_f_64_d_old"

sub_folders = os.listdir(samples_folder)
sub_folders = [name for name in sub_folders if not name.endswith(".txt")]

for index, name in enumerate(sub_folders):
    sub_folders[index] = os.path.join(samples_folder, name)

for name in sub_folders:
    new_folder = name.split('_old')[0] + name.split('_old')[1]
  
    os.makedirs(new_folder, exist_ok=True)
    existing_files = len(glob.glob(os.path.join(new_folder, "*isar.npy")))

    archivos_isarnpy = glob.glob(os.path.join(name, f"*isar.npy"))
    archivos_isartxt = glob.glob(os.path.join(name, f"*isar.txt"))
    archivos_matrizcomplexnpy = glob.glob(os.path.join(name, f"*complex.npy"))
    archivos_npy = glob.glob(os.path.join(name, f"*P.npy"))
    archivos_png = glob.glob(os.path.join(name, f"*P.png"))
    archivos_txt = glob.glob(os.path.join(name, f"*P.txt"))
    print(len(archivos_isarnpy))

    assert len(archivos_isarnpy) == len(archivos_isartxt) == len(archivos_matrizcomplexnpy) == len(archivos_npy) == len(archivos_png) == len(archivos_txt), f"Debe haber la misma cantidad en {name}"
    
    for file in range(len(archivos_isarnpy)):

        termination = archivos_isarnpy[file].split("sample_")[1].split("_isar")[0].split("_")

        dst_file_isarnpy = os.path.join(new_folder, f"sample_{existing_files+1}_{termination[1]}_{termination[2]}_isar.npy")
        dst_file_isartxt = os.path.join(new_folder, f"sample_{existing_files+1}_{termination[1]}_{termination[2]}_isar.txt")
        dst_file_matrizcomplexnpy = os.path.join(new_folder, f"sample_{existing_files+1}_{termination[1]}_{termination[2]}_matriz_complex.npy")
        dst_file_npy = os.path.join(new_folder, f"sample_{existing_files+1}_{termination[1]}_{termination[2]}.npy")
        dst_file_png = os.path.join(new_folder, f"sample_{existing_files+1}_{termination[1]}_{termination[2]}.png")
        dst_file_txt = os.path.join(new_folder, f"sample_{existing_files+1}_{termination[1]}_{termination[2]}.txt")

        shutil.copy(archivos_isarnpy[file], dst_file_isarnpy)
        shutil.copy(archivos_isartxt[file], dst_file_isartxt)
        shutil.copy(archivos_matrizcomplexnpy[file], dst_file_matrizcomplexnpy)
        shutil.copy(archivos_npy[file], dst_file_npy)
        shutil.copy(archivos_png[file], dst_file_png)
        shutil.copy(archivos_txt[file], dst_file_txt)

        existing_files +=1