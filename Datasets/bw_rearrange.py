import os
import glob
import numpy as np
import matplotlib.pyplot as plt

samples_folder = "/home/newfasant/N101-IA/Datasets/Raw/Samples_64_f_64_d_old"

sub_folders = os.listdir(samples_folder)

for index, name in enumerate(sub_folders):
    sub_folders[index] = os.path.join(samples_folder, name)

for name in sub_folders:
    archivos_isarnpy = glob.glob(os.path.join(name, f"*isar.npy"))

    for isar in archivos_isarnpy:
        matrix = np.load(isar)
        plt.xticks([])
        plt.yticks([])
        plt.imsave(isar.split('_isar')[0] + ".png", arr=matrix, cmap="gray", format="png")
        plt.close("all")