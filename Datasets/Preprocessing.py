import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys

def genera_numpys(carpeta_archivos_txt, carpeta_guardado_npy, w, wf, nd, nf, f0, scan_angle):
    """
    Generates .npy files from .txt files of field values and separates in 2 additional dimensions of real and imaginary parts.
    Moving between rows changes frequency, moving between columns changes angles.

    Args:
        carpeta_archivos_txt (string): Path to folder where .txt files are, from which information will be obtained.
        carpeta_guardado_npy (string): Path to folder where .npy files will be saved.
    """
    archivos_txt = glob.glob(os.path.join(carpeta_archivos_txt, f"*{scan_angle[0].upper()}.txt"))   # Extract .txt files from the specific folder.
    for archivo in archivos_txt:

        with open(archivo, 'r', encoding='utf-8') as f:
            lineas = f.readlines()              # Horizontal lines are splitted.
            if not lineas:
                continue    # If there is an empty file or line, the program continues.

            i = 0
            # Main matrix is created, and will be filled with field values, following the structure (#frequencies, #angles, 2(:=real & imag)).
            matriz = np.zeros(shape=(len(lineas), len(lineas[0].split(';'))-1, 2), dtype=np.float32)
            # DESCOMENTAR Y QUITAR LA SUPERIOR matriz = np.zeros(shape=(2, len(lineas), len(lineas[0].split(';'))-1), dtype=np.float32)

            for linea in lineas:
                partes = linea.split(';')               # Each line is separated into its components.
                j = 0
                # Each element of the line is being analized, and the last one is ommited because it follows after ";" character that end each line.
                for element in partes[:len(partes)-1]:  
                    nums = element.split(' ')               # 
                    matriz[i, j, 0] = float(nums[-2])       # Real and imaginary parts of the field are separated and consecuently saved in the matrix.
                    # DESCOMENTAR Y QUITAR LA SUPERIOR matriz[0, i, j] = float(nums[-2])       # Real and imaginary parts of the field are separated and consecuently saved in the matrix.
                    matriz[i, j, 1] = float(nums[-1])       #
                    # DESCOMENTAR Y QUITAR LA SUPERIOR matriz[1, i, j] = float(nums[-1])       #
                    j += 1
                i += 1
                
        nombre_salida = os.path.splitext(os.path.basename(archivo))[0] # 
        n_y, n_x = matriz.shape[0], matriz.shape[1]  # Number of frequencies and angles are extracted from the matrix.
        # DESCOMENTAR Y QUITAR LA SUPERIOR n_y, n_x = matriz.shape[1], matriz.shape[2]  # Number of frequencies and angles are extracted from the matrix.
        assert n_x == nd and n_y == nf, "The number of frequencies and angles in the matrix does not match the expected values."  # Check if the number of frequencies and angles is correct.
        
        lambda0 = 3e8 / f0          # Central wavelength is set as c/f_central
        
        delta_x = 3e8 / (2.0*2.0*wf)                        # The spacing of the vertical and horizontal axis is established
        delta_y = lambda0 / (2.0*2.0*np.radians(w))         #
        
        matriz_complex = matriz[:, :, 0] + 1j * matriz[:, :, 1]     # The matrix containing the full complex field value is created, as these values were stored separated in pairs
        # DESCOMENTAR Y QUITAR LA SUPERIOR matriz_complex = matriz[0, :, :] + 1j * matriz[1, :, :]     # The matrix containing the full complex field value is created, as these values were stored separated in pairs
        fft_general = np.fft.fftshift(np.fft.fft2(matriz_complex))         # The isar is made by doing the fft of that previous matrix
        general_template = np.zeros_like(matriz)
        general_template[:, :, 0] = fft_general.real
        general_template[:, :, 1] = fft_general.imag

        fft_perfil = np.fft.fftshift(np.fft.fft(a=matriz_complex, axis=0))
        perfil_template = np.zeros_like(matriz)
        perfil_template[:, :, 0] = fft_perfil.real
        perfil_template[:, :, 1] = fft_perfil.imag

        field_template = np.zeros_like(matriz)
        field_template[:, :, 0] = np.abs(matriz_complex)
        field_template[:, :, 1] = np.angle(matriz_complex)

        field_amp = np.zeros(shape=(matriz.shape[0], matriz.shape[1], 1))
        field_amp[:,:,0] = field_template[:,:,0]
        field_ph = np.zeros(shape=(matriz.shape[0], matriz.shape[1], 1))
        field_ph[:,:,0] = field_template[:,:,1]

        plt.xticks([])
        plt.yticks([])
        plt.imsave(os.path.join(carpeta_guardado_npy, nombre_salida) + "_field_amp.png", cmap='gray', arr=field_template[:,:,0], format="png")
        plt.close("all") # Close the figure to free memory

        plt.xticks([])
        plt.yticks([])
        plt.imsave(os.path.join(carpeta_guardado_npy, nombre_salida) + "_field_ph.png", cmap='gray', arr=field_template[:,:,1], format="png")
        plt.close("all") # Close the figure to free memory

        # colors = [
        # (0.0, 'black'),   # mínimo
        # (0.3, 'green'),   # bajo-medio
        # (0.6, 'yellow'),  # medio-alto
        # (1.0, 'red')      # máximo
        # ]

        # custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        # plt.imshow(np.abs(fft_general),cmap=custom_cmap, extent=[-n_x*delta_x/2, n_x*delta_x/2, -n_y*delta_y/2, n_y*delta_y/2])
        
        # Remover las etiquetas de los ejes
        plt.xticks([])
        plt.yticks([])

        # Show the colorbar in the figure
        # plt.colorbar()

        plt.imsave(os.path.join(carpeta_guardado_npy, nombre_salida) + ".png", arr=np.abs(fft_general), cmap="gray", format="png")
        # plt.savefig(os.path.join(carpeta_guardado_npy, nombre_salida) + ".png", bbox_inches='tight', pad_inches=0)
        plt.close("all") # Close the figure to free memory
        
        plt.xticks([])
        plt.yticks([])
        plt.imsave(os.path.join(carpeta_guardado_npy, nombre_salida) + "_perfil.png", arr=np.abs(fft_perfil), cmap="gray", format="png")
        plt.close("all") # Close the figure to free memory


        np.save(os.path.join(carpeta_guardado_npy, nombre_salida) + ".npy", matriz)          # The .npy file is saved in the desired root.
        np.save(os.path.join(carpeta_guardado_npy, nombre_salida) + "_matriz_complex.npy", matriz_complex)
        np.save(os.path.join(carpeta_guardado_npy, nombre_salida) + "_fft.npy", general_template)
        np.save(os.path.join(carpeta_guardado_npy, nombre_salida) + "_fft_perfil.npy", perfil_template)
        np.save(os.path.join(carpeta_guardado_npy, nombre_salida) + "_field.npy", field_template)
        np.save(os.path.join(carpeta_guardado_npy, nombre_salida) + "_field_amp.npy", field_amp)
        np.save(os.path.join(carpeta_guardado_npy, nombre_salida) + "_field_ph.npy", field_ph)
        np.savetxt(os.path.join(carpeta_guardado_npy, nombre_salida) + "_fft.txt", fft_general)
    
    print("Archivos npy e imágenes ISAR generados correctamente.")

def procesar_archivos(carpeta_archivos_out, carpeta_guardado_txt, scan_angle):
    """
    Generates .txt files from .out files of field values, moving between rows changes frequency, moving between columns changes angles.

    Args:
        carpeta_archivos_out (string): Path to folder where .out files are, from which information will be obtained.
        carpeta_guardado_txt (string): Path to folder where .txt files will be saved.
    """
    archivos_out = glob.glob(os.path.join(carpeta_archivos_out, "*.out"))   # Extract .out files from the specific folder.
    archivos_out.sort(key=lambda x: float((".".join(x.split('.')[:-1])).split('_')[-1]))

    datos_agrupados = {}
    # How many previous files are there?
    k = len(glob.glob(os.path.join(carpeta_guardado_txt, f"*T.txt"))) + len(glob.glob(os.path.join(carpeta_guardado_txt, f"*P.txt")))
    # Starting the tools that help checking if the fixed angle is theta (T) or phi (P).
    comparador = 0
    etiqueta = "P"
    key_ang = 0

    for archivo in archivos_out:
        
        with open(archivo, 'r', encoding='utf-8') as f:
            lineas = f.readlines()                  # Horizontal lines are splitted.
            if not lineas:
                continue  # Skip empty folders ie. with no lines.
            
            datos_por_archivo = {}

            # The following chunk of code analyzes which angle is not fixed in the scanning, between theta and phi, and saves the 
            # information consequently.
            for linea in lineas[1:4]:                       # It only analyzes the first lines of code because if one scanning has one
                partes = linea.strip().split()              # angle fixed, all scanning will behave equally.

                if comparador == 0:                         # If we are in the first line, gets the value of theta and stores it in clave_1.
                    clave_1 = partes[0]                     #
                if comparador == 1:                         # If we are in the second line, compares theta values and if they are the
                    if partes[0] == clave_1:                # same, the angle fixed is phi (P), otherwise is theta (T).
                        etiqueta = "P"                      # Also, to know which column has the angle that changes, the position of 
                        key_ang = 0                         # the column is stored in key_ang.
                    elif partes[0] != clave_1:
                        etiqueta = "T"
                        key_ang = 1
                comparador += 1

            for linea in lineas[1:]:  # Ignore the first line.
                partes = linea.strip().split()      # Each line is separated into is components, and first and last spaces are eliminated.                    

                if len(partes) >= 4:  # Making sure there is at least 4 columns
                    clave = partes[key_ang]                         # First column is theta, second is phi, and will be used as the key name of the sample.
                                                                    # To know which angle is fixed and will be in the sample name, key_ang is used.
                    real = partes[2]                                # Real and imaginary parts of the field are extracted.
                    imaginario = partes[3]                          #
                    if clave not in datos_por_archivo:              # If the key was not present before, the corresponding space is created.
                        datos_por_archivo[clave] = []               # in which the scanning information will be stored.
                    datos_por_archivo[clave].append(f"{real} {imaginario};")    # Field values are attached.
            
            for clave, valores in datos_por_archivo.items():            #
                if clave not in datos_agrupados:                        # The whole file structure is created joining information.
                    datos_agrupados[clave] = []                         # from different frecuencies.
                datos_agrupados[clave].append(" ".join(valores))        #

    for clave, lineas in datos_agrupados.items():
        k += 1
        nombre_salida = os.path.join(carpeta_guardado_txt, f"sample_{k}_{clave}_{etiqueta}.txt")
        # .txt files are saved in the corresponding place, depending on the sample number, fixed angle and T or P depending on what angle is not fixed.
        with open(nombre_salida, 'w', encoding='utf-8') as f:
            f.write("\n".join(lineas))
    
    print("Archivos txt generados correctamente.")

if __name__ == "__main__":
    carpeta_archivos_out = "./Nueva_carpeta"            # Path to folder where .out files will be used.
    carpeta_guardado_txt = carpeta_archivos_out         # Path to folder where .txt files will be saved, can be different.
    carpeta_guardado_npy = carpeta_archivos_out         # Path to folder where .npy files will be saved, can be different.
    # Calling the important partes of the workflow.
    procesar_archivos(carpeta_archivos_out, carpeta_guardado_txt)
    genera_numpys(carpeta_guardado_txt, carpeta_guardado_npy)
