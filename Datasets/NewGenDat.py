import argparse
import subprocess
import os
import time
import logging
from dir_and_freq_gen import generate_spherical_coordinates_file, generate_frequency_file
from Preprocessing import procesar_archivos, genera_numpys
from Reorganize_into_feature_vector_npy import reorganization, multiple_reorganization
from datetime import datetime

# Obtener la fecha y hora actual
actual_time = datetime.now()

userPath = os.getcwd().split('/')[2]
if userPath == "newfasant2":
    userPath = userPath + "/N101"

logs_folder_path=f'/home/{userPath}/N101-IA/Datasets/Logs'
os.makedirs(logs_folder_path, exist_ok=True)
logging.basicConfig(
    filename=f'/home/{userPath}/N101-IA/Datasets/Logs/Day_{actual_time.day}_{actual_time.month}_{actual_time.year}_Time_{actual_time.hour:02d}_{actual_time.minute:02d}_dataset.log',  # Name of the log file
    level=logging.INFO,  # Logs level (INFO, DEBUG, ERROR, etc.)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Procesar argumentos para enviarlos a los distintos programas necesarios.')
    parser.add_argument('-n', '--num_samples', type=int, help='Number of samples for each stl.')
    parser.add_argument('-g', '--geometries', nargs='+', type=str, help='Lista de archivos .msh')
    parser.add_argument('-o', '--output_path', type=str, default = f"/home/{userPath}/N101-IA/Datasets", help='Ruta terminada en /Datasets .')
    parser.add_argument('-f', '--freq_central', type=float, default=1e10, help='Valor de la frecuencia central del barrido.')
    parser.add_argument('--wf', type=float, default=2e9, help='Valor de la semianchura del barrido en frecuencias')
    parser.add_argument('--nf', type=int, default=64, help='Cantidad de frecuencias dentro del barrido.')
    parser.add_argument('-w', '--angular_width', type=float, default=20, help='Valor de la semianchura del barrido angular en grados.')
    parser.add_argument('--nd', type=int, help='Cantidad de ángulos contenidos en el ancho del barrido.')
    parser.add_argument('-d', '--scan_angle', type=str, default="theta", choices=["theta", "phi"], help='Elegir en qué ángulo se realizan los barridos.')
    parser.add_argument('-r', '--reorganize', action='store_true', help='Enable reorganization')
    parser.add_argument('--nr', nargs='+', type=int, help='Number of samples per geometrie to make up the reorganization folder')
    parser.add_argument('--pov', type=str, default=None, choices=["front", "left_side", "right_side", "back"], help='Which part of the object will be being looked')
    parser.add_argument('--cw', type=float, default=0, help='Value of the semi-width established for the pov cone in degrees')
    parser.add_argument('--snr', type=float, default=0, help='Value of the signal-to-noise ratio, expressed in dB, used to add noise to the dataset.')

    args = parser.parse_args()

    if args.pov ==None:
        assert args.cw == 0 , "If there's no POV, cone width must be 0"
    
    if args.pov == None:
        if args.snr == 0:
            top_folder_path = os.path.join(args.output_path, f"Raw/Samples_{args.nf}_f_{args.nd}_d")
        else:
            top_folder_path = os.path.join(args.output_path, f"Raw/Samples_{args.nf}_f_{args.nd}_d_SNR_{args.snr}")
    else:
        if args.snr == 0:
            top_folder_path = os.path.join(args.output_path, f"Raw/Samples_{args.nf}_f_{args.nd}_d_POV_{args.cw}")
        else:
            top_folder_path = os.path.join(args.output_path, f"Raw/Samples_{args.nf}_f_{args.nd}_d_POV_{args.cw}_SNR_{args.snr}")

    os.makedirs(top_folder_path, exist_ok=True)

    if args.num_samples != 0: # If the number of samples is 0, only the reorganization to obtain the classification dataset will be done. 

        out_folder_path = f"/home/{userPath}/N101-IA/Datasets/Raw/_Out"
        
        for item in os.listdir(out_folder_path):
            path = os.path.join(out_folder_path, item)
            os.remove(path)
        
        print(f'\nLos argumentos son {vars(args)}\n')
        logging.info(f'Los argumentos son {vars(args)}\n')
        print("Generation has started...\n")
        logging.info("Generation has started...\n")

        freq_file_name = os.path.join(top_folder_path, "frequencies.txt") # 
        spherical_coord_name = os.path.join(top_folder_path, "spherical_coordinates.txt")

        generate_frequency_file(args.freq_central, args.wf, args.nf, filename=freq_file_name)
        generate_spherical_coordinates_file(n=args.num_samples, angle=args.scan_angle, width=args.angular_width, m=args.nd, pov=args.pov, cone_width=args.cw, filename=spherical_coord_name)

        for geom in args.geometries: # Iterate over the desired geometries

            tic = time.time()

            msh_folder_path = os.path.join(args.output_path, "Geometries")
            msh_path = msh_folder_path + '/' + geom + ".msh"

            isar_aprox = args.nd
            print(f"The Isar Aprox is equal to {isar_aprox}\n")
            logging.info(f"The Isar Aprox is equal to {isar_aprox}")

            # Gemis generation starts for one geometry 
            result = subprocess.run(["python", f"/home/{userPath}/N101-IA/PostGIS/NewFasant-PostGIS/main.py", 
                                    "-m", "GO-PO", 
                                    "-o", f"/home/{userPath}/N101-IA/Datasets/Raw/_Out/Sample.out",
                                    "-c", "44",
                                    "-e","1",
                                    "-d", spherical_coord_name, 
                                    "-f", freq_file_name, 
                                    "-g", msh_path,
                                    "1", "1",
                                    "-a",
                                    "--isar", str(isar_aprox)
                                    ]
                                    , capture_output=True, text=True)

            # print(" ".join(["python", f"/home/{userPath}/N101-IA/PostGIS/NewFasant-PostGIS/main.py", 
            #                         "-m", "GO-PO", 
            #                         "-d", spherical_coord_name, 
            #                         "-f", freq_file_name, 
            #                         "-g", msh_path,
            #                         "1", "1",
            #                         "-a",
            #                         # "--isar", str(isar_aprox)
            #                         ]))
            
            print(result.stdout,"\n")
            logging.info(result.stdout)

            print(result.stderr,"\n")
            logging.info(result.stderr)

            sample_folder_path = top_folder_path + '/' + geom
            os.makedirs(sample_folder_path, exist_ok = True)

            toc_1 = time.time()

            print(f"\nGEMIS generation of {args.num_samples} samples of the {geom} completed in {toc_1-tic} seconds, {(toc_1-tic)/60} minutes.\n")
            logging.info(f"GEMIS generation of {args.num_samples} samples of the {geom} completed in {toc_1-tic} seconds, {(toc_1-tic)/60} minutes.")

            procesar_archivos(f"/home/{userPath}/N101-IA/Datasets/Raw/_Out", sample_folder_path, args.scan_angle)
            genera_numpys(sample_folder_path, sample_folder_path, args.angular_width, args.wf, args.nd, args.nf, args.freq_central, args.scan_angle, args.snr)

            toc_2 = time.time()

            print(f"\nPreprocessing of {args.num_samples} samples of the {geom} completed in {toc_2-toc_1} seconds, {(toc_2-toc_1)/60} minutes.\n")
            logging.info(f"Preprocessing of {args.num_samples} samples of the {geom} completed in {toc_2-toc_1} seconds, {(toc_2-toc_1)/60} minutes.")
    else:
        print("\nNo samples were generated.\n")
        logging.info("\nNo samples were generated.\n")

    if args.reorganize: # If reorganization is enabled (-r appears in the command line)
        print(f"\nReorganization of samples has started...\n")
        logging.info(f"\nReorganization of samples has started...\n")

        nr_dictionary = multiple_reorganization(args.nr, args.geometries)
        reorganization(args.scan_angle, nr_dictionary, args.nf, args.nd, args.output_path, top_folder_path, args.cw, args.snr) # Reorganize the samples

        print(f"\nReorganization of samples completed\n")
        logging.info(f"\nReorganization of samples completed\n")

if __name__ == '__main__':
    main()

# python NewGenDat.py -g 2_cubes_zero_two_fixed -n 3 --nd 64 --nf 64 -w 1.72 -d "phi" -f 1e10 --wf 3e8 
# vapor55_50000_real Scaneagle_UAV_50000_real Drone_X8_quadrocopter_50000_real Agriculteur_UAV_real 
