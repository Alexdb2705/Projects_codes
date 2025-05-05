import argparse
import sys, os
import subprocess
from pathlib            import Path
import time
import multiprocessing as mp

command_name        = "prueba"
command_description = "Piramides en Blender"

try:
    import bpy, bmesh
    from mathutils import Matrix, Euler, Vector
except ModuleNotFoundError:
    print("Este script solo se puede ejecutar desde blender.")
    # exit(0)
import random
import math
from stl import mesh
import numpy as np
import re
import shutil

####################### Utility function for the subcommand structure ########################################

def add_subparser(subparsers):
    """
        add_subparser:: ArgParse -> ArgParse
        Add asubparser for command line switches need for this very module to run properly
        An additional "--test" command line switch allows to perform doctest tests from the main program for this module
    """
    from library.file_checks    import check_file_arg
    is_reachable = lambda s: check_file_arg(                                    # lambda for file checking
                 argparse.ArgumentTypeError(f"'{s}' no encontrado o accesible!")# exception to raise
                ,Path(s))                                                       # Path to check

    command_parser              = subparsers.add_parser(command_name, description = command_description)
    ### Additional command line arguments needed for this puzzle like data files etc..
    command_parser.add_argument("--test", help = f"test suite del módulo {__name__}", action='store_true') # optional built-in test suite switch

    if not '--test' in sys.argv:                                                # if not '--test' requested show additional params
        command_parser.add_argument('-a', '--background',  action=argparse.BooleanOptionalAction, default=False,
                                    help = "¿Se está ejecutando con el python de Blender?")
        command_parser.add_argument('-b', '--blender',      type = str,  default = r"C:\Program Files\Blender Foundation\Blender 4.2",
                                    help = "Ruta de la carpeta donde está blender.exe")
        command_parser.add_argument('-v', '--version',   type = str,  default = "4.2",  
                                    help = "Versión del Blender")
        command_parser.add_argument('-i', '--img',   type = str,    default = os.path.join("C:\\Users", os.getlogin(), "Desktop", "Img"),
                                    help = "Ruta de la carpeta donde se guardarán las imagenes generadas")
        command_parser.add_argument('-n', '--num',   type = int,  default = 20, 
                                    help = "Número de ejemplos que se crean")
        command_parser.add_argument('-s', '--stl',   type = str,  default = os.path.join("C:\\Users", os.getlogin(), "Desktop", "geom.stl"),
                                    help = "Ruta del archivo .stl que contiene la geometría deseada")
        command_parser.add_argument('-r', '--rot_path',   type = str,  default = os.path.join("C:\\Users", os.getlogin(), "Desktop", "Rot"),
                                    help = "Ruta de la carpeta donde esta guardado los archivos con las matrices de rotación y los ángulos esféricos")
        command_parser.add_argument('--NFpath', type = str,  default = os.path.join("C:\\Users", os.getlogin(), "Desktop", "NF"),
                                    help = "Folders path for the NewFasant simulation")
        command_parser.add_argument('--NProcs', type = str,  default = 6,
                                    help = "")
        command_parser.add_argument('--BW', type = str,  default = 5,
                                    help = "Band Width in degrees (crossrange), Ancho de banda en grados (crossrange)")
        command_parser.add_argument('--nBW', type = str,  default = 128,
                                    help = "Número de muestras en crossrange")
        command_parser.add_argument('--f0', type = str,  default = 10,
                                    help = "Frecuencia central en Ghz")
        command_parser.add_argument('--fBW', type = str,  default = 1,
                                    help = "Ancho de banda en frecuencia GHz (range), Band Width in frecuency GHz (range)")
        command_parser.add_argument('--nfBW', type = str,  default = 128,
                                    help = "Número de muestras en frecuencia, Number of samples in frecuency")

    ### Additional command line arguments needed for this puzzle
    
    command_parser.set_defaults(module = sys.modules[__name__])                 # export this module module to run
    
    return subparsers

################################# Puzzle Functions ##########################################

def main(args):
    """
    """
    
    if not args.background:
        print(command_description)
        print("")
        print("ID del proceso del main(): ", os.getpid())

        gen_angles_and_matrices(args.num, args.rot_path)

        blender_path = os.path.join(args.blender, args.version, "python", "bin", "python.exe")
        target_path = os.path.join(Path(os.path.dirname(__file__)).parent, "main.py")

        pool = mp.Pool(2)
        params = [(1, blender_path, target_path, args.img, args.num, args.stl, args.rot_path),
                  (2, args.img, args.stl, args.rot_path, args.NFpath, args.num, args.NProcs, args.nBW, args.f0, args.nfBW)]
        
        for _ in pool.imap_unordered(parallel_execution, params):
            continue

        pool.close()
        pool.join()
        
        return None
    else:
        # Estamos trabajando desde el python de blender. Si no tenemos sus paquetes, hemos acabado
        blender_creation(args.img, args.stl, args.rot_path, args.num)

def parallel_execution(params):
    if params[0] == 1:
        trigger_blender(*params[1:])
    else:
        NewFasant_creation(*params[1:])

def trigger_blender(blender_path, target_path, img, num, stl, rot_path):
    print(" ".join([blender_path, 
                    target_path,
                    "prueba",
                    "--background",                   # Solo si se ejecuta desde blender
                    # # "-P",
                    # "--",                        
                    "-i",   img,
                    "-n",   str(num),
                    "-s",  stl,
                    "r",  rot_path
                    ]))
    subprocess.run([blender_path, 
                    target_path,
                    "prueba",
                    "--background",                   # Solo si se ejecuta desde blender
                    # # "-P",
                    # "--",                        
                    "-i",   img,
                    "-n",   str(num),
                    "-s",  stl,
                    "-r",  rot_path
                    ])   

# --------------------------- Funciones para la simulación de Blender ------------------------------------

# Función para pasar de coordenadas cartesianas a los ángulos polar (theta), y azimutal (phi)
def cartesian_to_spherical(x):
    theta = (np.arccos(x[2] / np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)))*180/np.pi         # Obtención del ángulo theta, y conversión de radianes a grados                                                                            # Obtención del ángulo phi, y conversión de radianes a grados
    if x[0]==0:                                                                        # Si la coordenada x es cero, la función no estaría definida y el resultado entonces debe ser éste
        phi = (np.pi/2*np.sign(x[1]))*180/np.pi
    else:
        phi = (np.arctan2(x[1], x[0]) + np.pi)*180/np.pi
    return theta, phi

# Función para rotar el objeto con una inclinación aleatoria
def convert_spherical():
    # Se generan los ángulos aleatorios de rotación alrededor de cada eje
    theta_al_x = np.random.random() * 2 * np.pi                                                         
    theta_al_y = np.random.random() * 2 * np.pi                                                         
    theta_al_z = np.random.random() * 2 * np.pi         

    # Funciones trigonométricas que irán dentro de las matrices de rotación
    cos_al_x = np.cos(theta_al_x)                                                                       
    sin_al_x = np.sin(theta_al_x)
    cos_al_y = np.cos(theta_al_y)
    sin_al_y = np.sin(theta_al_y)
    cos_al_z = np.cos(theta_al_z)
    sin_al_z = np.sin(theta_al_z)

    # Matrices de rotación alrededor de cada uno de los dos ejes
    R_x = np.array([[1, 0, 0], [0, cos_al_x, -sin_al_x], [0, sin_al_x, cos_al_x]], dtype = np.float32)
    R_y = np.array([[cos_al_y, 0, sin_al_y], [0, 1, 0], [-sin_al_y, 0, cos_al_y]], dtype = np.float32)
    R_z = np.array([[cos_al_z, -sin_al_z, 0], [sin_al_z, cos_al_z, 0], [0, 0, 1]], dtype = np.float32)

    # Se hace un orden aleatorio de composición de rotaciones y se calcula la rotación final
    list_matr = [R_x, R_y, R_z]
    np.random.shuffle(list_matr)
    rotation_matrix = np.matmul(np.matmul(list_matr[0], list_matr[1]), list_matr[2])

    return np.array(rotation_matrix)

# Función que genera archivos en los que se guarda una lista con las matrices de rotación, y otro en el que se guardan parejas de ángulos theta, phi correspondientes a las posiciones de las cámaras tras rotarse 
def gen_angles_and_matrices(num_samples, rot_path):

    os.makedirs(rot_path, exist_ok = True)
    np.set_printoptions(suppress=True)

    # Guardamos los resultados previos y añadimos los que tenemos
    if os.path.exists(os.path.join(rot_path, "spherical_angles.npy")):                           # Se comprueba si en esa ruta ya hay un archvio previo, para escribir los casos nuevos a continuación y no sobrescribir
        matrix_angles = np.load(os.path.join(rot_path, "spherical_angles.npy"))                  # Cargamos el archivo previo ya existente en la ruta especificada
        # matrix_angles = matrix_angles['arr_0']                                                 # Asegurarse de que la matriz tiene la forma correcta cuando el csv solo contiene una iteracion         
        new_matrix_angles = np.concatenate((matrix_angles, np.zeros((num_samples, 7))), axis=0)  # Se deja el hueco, con todo 0, para los casos nuevos que se van a crear esta vez
        previous_samples = matrix_angles.shape[0]                                                # Número de ejemplos previos
    
    else:
        new_matrix_angles = np.zeros((num_samples,7), dtype=np.float32)                          # Si no hay un archivo previo, se crea la matriz vacía que tras rellenarse se guardará en el archivo nuevo
        previous_samples = 0                                                                     # Número de ejemplos previos, pues al no poder leerse de ningún archivo se establece así      

    if os.path.exists(os.path.join(rot_path, "rot_matrices.npz")):                               # Se comprueba si en esa ruta ya hay un archvio previo, para escribir los casos nuevos a continuación y no sobrescribir
        loaded_npz = np.load(os.path.join(rot_path, "rot_matrices.npz"))                         # Cargamos el archivo previo ya existente en la ruta especificada
        rot_matrices = [loaded_npz[f'arr_{i}'] for i in range(len(loaded_npz.files))]            # Asegurarse de que la matriz tiene la forma correcta cuando el csv solo contiene una iteracion         
    
    else:
        rot_matrices=[]                                                                          # Si no hay casos previos, se genera la lista vacía que irá rellenándose a continuación

    # Bucle que se genera un número de veces igual al número de ejemplos nuevos que se quieran tener
    for j in range(num_samples):
        rot_matrix = convert_spherical()                                                                    # Se crea la matriz de rotación
        rot_matrices.append(rot_matrix)                                                                     # Se añade la matriz de rotación a la lista de matrices anterior
        cameras_positions_rot = np.matmul(np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]]), rot_matrix.T)     # Se obtiene la posición de las cámaras rotadas utilizando para ello la matriz de rotación inversa a la de la pirámide (traspuesta al ser de rotación)
        angles = np.zeros((1,6))                                                                            # Se genera el array vacío en el que se guardarán las tres parejas de ángulos de cada ejemplo

        for i in range(3):
            angles[0,2*i:2*i+2] = cartesian_to_spherical(cameras_positions_rot[i,:])                        # Se rellena el array de ángulos con los correspondientes en cada ejemplo
        new_matrix_angles[previous_samples + j, 0] = previous_samples + j + 1                               # En la primera columna se guarda el número de ejemplo o índice
        new_matrix_angles[previous_samples + j, 1:] = angles                                                # En las otras columnas se guardan los 6 ángulos que se han establecido anteriormente
    
    angles_npz_path = os.path.join(rot_path, "spherical_angles.npy")                                        # Se establecen las rutas de los archivos de matrices y ángulos
    rot_matrices_npz_path = os.path.join(rot_path, "rot_matrices.npz")                                       
    np.save(angles_npz_path, new_matrix_angles)                                                             # Se guardan los ángulos en el archivo y ruta correspondiente 
    np.savez(rot_matrices_npz_path, *rot_matrices)                                                           # Se guardan las matrices de rotación, desempaquetando las matrices para guardarlas individualmente

# Función para renderizar la imagen desde diferentes ángulos
def render_from_view(axis, rotation_matrix, image_path, dist_max):
    
    # Configurar la cámara
    camera = bpy.data.objects.get("Camera") 
    light = bpy.data.objects.get("Point")       
    
    # Localizacion inicial de la camara                                                                               # Se inicia un contador de tiempo para calcular cuánto tardan determinadas secciones del programa
    camera.location = np.array((0, 0, 6 * dist_max) if axis == 'z' else (6 * dist_max, 0, 0) if axis == 'x' else (0, 6 * dist_max, 0))
    
    # Rotamos la perspectiva de la cámara para que mire al objeto, en función del eje en el que se encuentre 
    if axis == 'x':
        track_to_simulator = Euler((math.pi/2, 0, math.pi/2), 'XYZ')
    elif axis == 'y':
        track_to_simulator = Euler((math.pi/2, 0, math.pi), 'XYZ')
    else:
        track_to_simulator = Euler((0,0,0), 'XYZ')

    camera.rotation_euler = track_to_simulator                                                               # Lo establecemos con esa rotación, para hacer a mano lo que Blender establecería con el track-to
    
    # Rotamos la camara
    camera.location = np.matmul(camera.location, np.array(rotation_matrix).T)
    # Rotamos en consecuencia la perspectiva de la camara
    camera.delta_rotation_euler = Matrix(rotation_matrix).to_euler('XYZ')
    
    # Configurar la cámara activa
    bpy.context.scene.camera = camera
    
    light.location = camera.location
    
    r = bpy.context.scene.render

    # Configurar la ruta de salida y tamaño de la imagen
    r.filepath = image_path
    r.resolution_x = 1080
    r.resolution_y = 1080
    
    # Renderizar
    bpy.ops.render.render(write_still=True)


# ---------------------------- Funciones para la simulación de NewFasant ------------------------------------

def datos_calculate():
    # definicion de dicciona
    datos_dict = {
        "line1": "******************",
        "line2": "M",
        "line3": "R",
        "line4": "******************",
        "line5": "10.0",
        "line6": "11.0",
        "line7": "16",
        "line8": "******************",
        "line9": "F",
        "line10": "90.0",
        "line11": "90.0",
        "line12": "1",
        "line13": "0.0",
        "line14": "6.0",
        "line15": "1",
        "line16": "******************",
        "line17": "\"data/geom.fc\"",
        "line18": "\"result/resul\"",
        "line19": "******************",
        "line20": "S",
        "line21": "1",
        "line22": "S",
        "line23": "******************",
        "line24": "N",
        "line25": "******************",
        "line26": "N",
        "line27": "N",
        "line28": "**** Datos SpeedUp ****",
        "line29": "N",
        "line30": "-1.0"
    }
    return datos_dict

def datos_mesh():
    # definicion de diccionario mesh
    datos_dict = {
        "line1": "\"./data/geom.nur\"",
        "line2": "1.0E10",
        "line3": "2.0",
        "line4": "2.0",
        "line5": "F",
        "line6": "F",
        "line7": "F",
        "line8": "F",
        "line9": "F",
        "line10": "T",
        "line11": "F",
        "line12": "5",
        "line13": "10.0",
        "line14": "1.0E-6",
        "line15": "F",
        "line16": "F",
        "line17": "1",
        "line18": "1",
        "line19": "F",
        "line20": "0",
        "line21": "2",
        "line22": "0",
        "line23": "F",
        "line24": "F",
        "line25": "F",
        "line26": "0",
        "line27": "F",
        "line28": "0 0 0 0 0 0",
        "line29": "0",
        "line30": "-1",
        "line31": "10",
        "line32": "0",
        "line33": "0.0 0.0 0.0"
    }
    return datos_dict

def escribir_datos_dat(calculate_folder, mi_diccionario):
    # Esta función escribe el datos_dat de calcular
    # Escribir el diccionario a un archivo
    with open(calculate_folder, 'w') as archivo:
        for clave, valor in mi_diccionario.items():
            archivo.write(f"{valor}\n")
        archivo.close()

def escribir_geom_nurbs_from_stl(your_mesh, mesh_folder):        
    # Esta función escribe el fichero de geometría .nur para el stl
    # Escribir el formato .nur de mesh
    with open(mesh_folder, 'w') as archivo:
        archivo.write("1\n")
        archivo.write("1\n")
        archivo.write("OBJECT\n")
        archivo.write("MESH_SURFACE\n")
        archivo.write("F\n")
        archivo.write("F\n")
        archivo.write("4\n")
        archivo.write(str(len(your_mesh)*3)+"\n")
        archivo.write(str(len(your_mesh))+"\n")
        # Generate vertex of every element
        for element in your_mesh:
            archivo.write(f'{element[0]} {element[1]} {element[2]}\n')
            archivo.write(f'{element[3]} {element[4]} {element[5]}\n')
            archivo.write(f'{element[6]} {element[7]} {element[8]}\n')
        # Generate vertex references for every element
        for i in range(0, len(your_mesh)):
            archivo.write(f'{(i)*3} {(i)*3 + 1} {(i)*3 + 2} {(i)*3 + 2}\n')
        archivo.close()

def copy_and_replace(source_path, destination_path):
    if os.path.exists(destination_path):
        os.remove(destination_path)
    shutil.copy2(source_path, destination_path)

def dim_calculator(stl_path, nBW_frecuencia, nBW_grados, f0):
    mesh_for_verts = mesh.Mesh.from_file(stl_path)
    verts = np.around(np.unique(mesh_for_verts.vectors.reshape([int(mesh_for_verts.vectors.size/3), 3]), axis=0),6)
    dist_max = 0
    for i in range(len(verts)):
        if sum(verts[i] ** 2) > dist_max:
            dist_max = sum(verts[i] ** 2)
    dist_max = np.sqrt(dist_max)
    print("dist_max= " , dist_max)
    BW_Hz = nBW_frecuencia * (299792458 / (2 * 2 * dist_max))
    BW_rad= nBW_grados * ((299792458/(f0*10**9)) / (2 * 2 * dist_max))
    print("drange=", 2*dist_max)
    print("Incremento=", 2*dist_max/nBW_frecuencia)
    return BW_Hz , BW_rad
    
# ---------------------------- Funciones principales ------------------------------------

# Crea un cierto numero de piramides en una cierta carpeta
def blender_creation(directory_gen, stl_path, rot_path, num_samples):
    tic = time.time()
    # Creación de directorios, si no existen ya 
    os.makedirs(directory_gen, exist_ok = True)
    directory_path_x = os.path.join(directory_gen, "Img_axis_x")
    directory_path_y = os.path.join(directory_gen, "Img_axis_y")
    directory_path_z = os.path.join(directory_gen, "Img_axis_z")
    os.makedirs(directory_path_x, exist_ok = True)
    os.makedirs(directory_path_y, exist_ok = True)
    os.makedirs(directory_path_z, exist_ok = True)

    np.set_printoptions(suppress=True)

    # Eliminamos un escenario si lo hubiera
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Extraemos la información de los vértices del archivo stl, y la cantidad de ellos que tiene la geometría a tratar en cuestión
    mesh_for_verts = mesh.Mesh.from_file(stl_path)
    verts = np.around(np.unique(mesh_for_verts.vectors.reshape([int(mesh_for_verts.vectors.size/3), 3]), axis=0),6)
    num_verts = verts.shape[0]
    print("Number of vertices are", num_verts)
    dist_max = 0
    for i in range(len(verts)):
        if sum(verts[i] ** 2) > dist_max:
            dist_max = sum(verts[i] ** 2)
    dist_max = np.sqrt(dist_max)

    # Cargamos las matrices de rotación usadas para cada pirámide

    loaded_npz = np.load(os.path.join(rot_path, "rot_matrices.npz"))
    rotation_matrices = [loaded_npz[f'arr_{i}'] for i in range(len(loaded_npz.files))]
    rotation_matrices = rotation_matrices[-num_samples:]  # Nos quedamos con las matrices de rotación que necesitamos para añadir las n nuevas

    # Comprobar que la longitud de la lista de matrices de rotación es la del número de ejemplos, y si no, mostrar el error a continuación
    assert len(rotation_matrices) == num_samples, "El número de matrices de rotación no coincide con el número de muestras"

    # Guardamos los resultados previos y añadimos los que tenemos
    if os.path.exists(os.path.join(directory_gen, "coords.npy")):
        matrix = np.load(os.path.join(directory_gen, "coords.npy"))         # Matriz cuya primera linea es el ID y las otras tres son las coordenadas
        # matrix = matrix.reshape((-1, 1 + num_verts*3))                                     # Asegurarse de que la matriz tiene la forma correcta cuando el csv solo contiene una iteracion         
        new_matrix = np.concatenate((matrix, np.zeros((num_samples, 1 + num_verts*3))), axis=0)    # Se juntan, por las filas, las coordenadas ya existentes con una matriz de ceros que se rellenará en cada nuevo ejemplo
        previous_samples = matrix.shape[0]                                  # Número de ejemplos previos
        max_name = np.max(matrix[:,0])                                      # Número de ID máximo
    else:
        new_matrix = np.zeros((num_samples, 1 + num_verts*3), dtype=np.float32)     # Si no hay un archivo con coordenadas anterior, se genera una matriz de ceros que se rellenará con los nuevos casos y pasará a ser la que se guarde
        previous_samples = 0                                                # Número de ejemplos previos
        max_name = 0                                                        # Número de ID máximo
    
    bpy.ops.wm.stl_import(filepath=stl_path)
    geometry_obj = bpy.data.objects[-1]
    bpy.context.view_layer.objects.active = geometry_obj

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.shade_flat() 
    bpy.ops.object.select_all(action='DESELECT')

    geometry_obj.select_set(True)
    
    # Asegurarse de que esté centrado en el origen
    geometry_obj.location = (0, 0, 0)

    bpy.ops.object.camera_add()
    bpy.ops.object.light_add(type='POINT')
    light = bpy.data.objects.get("Point") 
    light.data.energy = 1000 * dist_max ** 2  # Ajusta la intensidad de la luz según sea necesario
    # Agregar el modificador 'Track To' para que la luz apunte al centro de la pirámide
    light_constraint = light.constraints.new(type='TRACK_TO')
    light_constraint.target = geometry_obj # Usar la geometría como objetivo
    light_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    light_constraint.up_axis = 'UP_Y'

    # Comenzamos a generar, se recorre el bucle una vez por cada ejemplo nuevo que se quiera crear
    for j in range(num_samples):

        # La matriz de rotación para cada ejemplo vendrá dada por el caso correspondiente en la lista de matrices ya cargada anteriormente
        rotation_matrix = rotation_matrices[j]

        # Obtener la posición de los vértices después de la rotación, (fila * matriz, para dar otro vector fila)
        verts_after_rotation = np.matmul(verts, rotation_matrix)

        # Generacion de los path
        num_name = int(j + max_name + 1)                                 # ID del sample nuevo que se va a añadir, obteniéndose como el máximo prexistente más el número de ejemplos ya generados en este caso. (+1 al iniciarse el bucle en j=0)
        # Se recorre, por parejas, los directorios de los ejes, y el eje en sí
        for path_axis, axis in zip([directory_path_x, directory_path_y, directory_path_z],['x', 'y', 'z']): 
            # Renderizar imágenes y almacenar las rutas de las imágenes
            image_path = os.path.join(path_axis, f"sample_{num_name}_{axis}.png")
            render_from_view(axis, rotation_matrix, image_path, dist_max)
        
        # Guardamos los datos
        new_matrix[previous_samples + j, 0] = num_name                      # ID de la imagen
        # Para cada vértice distinto de la geometría
        for i in range(num_verts):
            new_matrix[previous_samples + j, (1+i*3):((4+i*3))] = verts_after_rotation[i,:]        # Coordenadas
    
    # Guardar el dataframe en un archivo CSV
    np.save(os.path.join(directory_gen, "coords.npy"), new_matrix)
    np.savetxt(os.path.join(directory_gen, "coords.csv"), new_matrix, delimiter=';')

    toc=time.time()
    print("Blender has taken ",toc-tic," seconds.")

def NewFasant_creation(directory_gen, stl_path, rot_path, NF_folders_path, num_samples,
                       nprocs, nBW_grados, f0, nBW_frecuencia):
    
    tic = time.time()

    import numpy as np
    import matplotlib.pyplot as plt

    BW_Hz, BW_rad = dim_calculator(stl_path, nBW_frecuencia, nBW_grados, f0)
    BW_GHz = BW_Hz/10**9
    BW_grados = BW_rad * 180/np.pi

    # print(f"Band Width in GHz: {BW_GHz}", f"Band Width in degrees: {BW_grados}\n", BW_Hz, BW_rad)

    print("ID del proceso del NewFasant_creation: ", os.getpid())

    os.makedirs(os.path.join(directory_gen, "ISAR"), exist_ok = True)
    os.makedirs(os.path.join(directory_gen, "NPY"), exist_ok = True)

    your_mesh = mesh.Mesh.from_file(stl_path)
    escribir_geom_nurbs_from_stl(your_mesh, os.path.join(NF_folders_path, "mesh_folder_PO") + "\\data\\geom.nur")

    # Generación del datos_mallador.dat de mallar
    datos_mesh_dict = datos_mesh()
    escribir_datos_dat(os.path.join(NF_folders_path, "mesh_folder_PO") + "\\data\\datos_mallador.dat", datos_mesh_dict)

    # Comando para mallar. Fichero feko debe estar acon ruta completa al directrio de mallado
    comando_mesh = f"mpiexec -np {nprocs} newfasant_mesh {os.path.join(NF_folders_path, 'mesh_folder_PO')}\\feko"  

    # Ejecutar el comando y capturar la salida
    po = subprocess.run(comando_mesh, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Mostrar la salida y posibles errores
    # print("Salida:")
    # print(po.stdout)
    # print("Errores:")
    # print(po.stderr)

    # ---------------------------------------- Copiar ficheros -------------------------------------
    
    mesh_path = os.path.join(NF_folders_path, "mesh_folder_PO", "data")
    calculate_path = os.path.join(NF_folders_path, "calculate_folder_PO", "data")

    # Copiar ficheros de mallado
    files_list = ["facets_duplicated.dat",
                  "geom.fc",
                  "geom.msh",
                  "materials_fc.dat",
                  "origpar.dat",
                  "grupos_efie.dat",
                  "materials.dat",
                  "objetos.dat",
                  "geom.nur"
                  ]

    for file in files_list:
        copy_and_replace(os.path.join(mesh_path, file), os.path.join(calculate_path, file))

    # Cargar ángulos
    angulos = np.load(os.path.join(rot_path, "spherical_angles.npy"))
    # Seleccionar los ángulos que corresponden a los ejemplos que se van a simular, sin sobreescribir datos e imagenes que puedan haberse generado previamente
    angulos = angulos[-num_samples:]
    previous_samples = angulos.shape[0]-num_samples
    datos_calculate_dict = datos_calculate()

    # Barrido en phi
    datos_calculate_dict["line9"] = "F"
    # Barrido de frecuencias
    datos_calculate_dict["line5"] = str(f0-BW_GHz/2.0)
    datos_calculate_dict["line6"] = str(f0+BW_GHz/2.0)
    datos_calculate_dict["line7"] = str(nBW_frecuencia)
    # Ciclo al número de ángulos
    for element in angulos:
        tic_0 = time.time()
        for j in range(1,4):

            tic_in = time.time()

            theta = element[1+(j-1)*2]
            phi = element[2+(j-1)*2]
            
            # modificación de datos de barrido
            datos_calculate_dict["line10"]= str(phi-BW_grados/2.0)
            datos_calculate_dict["line11"]= str(phi+BW_grados/2.0)
            datos_calculate_dict["line12"]= str(nBW_grados)
            datos_calculate_dict["line13"]= str(theta)
            datos_calculate_dict["line14"]= str(theta)

            escribir_datos_dat(os.path.join(NF_folders_path, "calculate_folder_PO") + "\\data\\data.dat", datos_calculate_dict)

            # Comando para simular. Fichero feko debe estar acon ruta completa al directrio de mallado
            comando_mesh = f"mpiexec -np {nprocs} newfasant_po {os.path.join(NF_folders_path, 'calculate_folder_PO')}\\feko"  

            # Ejecutar el comando y capturar la salida
            po = subprocess.run(comando_mesh, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Mostrar la salida y posibles errores
            # print("Salida:")
            # print(po.stdout)
            # print("Errores:")
            # print(po.stderr)

            # Se lee el fichero de resultados

            f=open(f"{os.path.join(NF_folders_path, 'calculate_folder_PO')}\\result\\RcsFieldTO.out","r+")
            profile = []
            freq = []
            theta = []
            phi = []
            campo = []
            cont = 0
            for line in f:
                line=line.split()
                if line:
                    if "#FREQUENCY" in str(line[0]):
                        freq.append(float(line[2]))
                    elif "THETA" in str(line[0]):
                        if cont > 1:
                            campo.append(profile)
                            profile=[]
                        cont = 0                
                    elif len(line) == 10:
                        if line[0] not in theta:
                            theta.append(line[0])
                        if line[1] not in phi:
                            phi.append(line[1])
                        profile.append(complex(float(line[2]),float(line[3]))) # Theta theta field
                        cont = cont + 1
            f.close()
            if cont == 1:
                campo = profile

            # Se hace la fft 1D o 2D con numpy

            deltax = 3E8/(2.0*(freq[-1]-freq[0]))
            nx = len(freq)

            if cont ==1 : # Prefil 1D fft
                perfil = np.fft.fftshift(np.fft.fft(campo))
                plt.plot(np.arange(nx*deltax/2, -nx*deltax/2, -deltax), np.abs(perfil))

            else:

                isar = np.fft.fftshift(np.fft.fft2(campo))
                f0 = np.mean(freq)
                lamda0 = 3E8/f0

                if len(theta) > 1:
                    ny = len(theta)
                    deltay = lamda0/(2.0*np.radians(float(theta[-1])-float(theta[0])))
                elif len(phi) > 1:
                    ny = len(phi)
                    deltay = lamda0/(2.0*np.radians(float(phi[-1])-float(phi[0])))
                
                plt.imshow(np.abs(isar),extent=[-ny*deltay/2, ny*deltay/2, -nx*deltax/2, nx*deltax/2])
                # # Remover las etiquetas de los ejes
                # plt.xticks([])
                # plt.yticks([])
                plt.savefig(f'{os.path.join(NF_folders_path, "calculate_folder_PO")}\\result\\isar.png', bbox_inches='tight', pad_inches=0)
                # plt.show()
                plt.close()

            #copy output files

            np.save(f'{os.path.join(directory_gen, "NPY")}\\{int(element[0]) + previous_samples}_{j}.npy', isar)
            shutil.copy(f'{os.path.join(NF_folders_path, "calculate_folder_PO")}\\result\\isar.png',f'{os.path.join(directory_gen, "ISAR")}\\isar_{int(element[0]) + previous_samples}_{j}.png')
            shutil.copy(f'{os.path.join(NF_folders_path, "calculate_folder_PO")}\\result\\RcsFieldTO.out',f'{os.path.join(directory_gen, "ISAR")}\\RcsFieldTO_{int(element[0] + previous_samples)}_{j}.txt')
            
            toc_in = time.time()
            print(f"\n #### Tiempo de ejecución iteración {int(element[0])}_{j}: {toc_in - tic_in} #### \n")

            del profile
            del freq
            del theta
            del phi
            del campo
        
        toc_0 = time.time()
        print(f"\n #### Tiempo ejecución sample {int(element[0])}: {toc_0 - tic_0} #### \n")
    
    toc = time.time()
    print("NewFasant has taken ",toc-tic," seconds.")
