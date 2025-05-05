import argparse
import sys, os
import subprocess
from pathlib            import Path
import time

command_name        = "prueba"
command_description = "Piramides en Blender"

try:
    import bpy, bmesh
    from mathutils import Matrix, Euler, Vector
except ModuleNotFoundError:
    print("Este script solo se puede ejecutar desde blender.")
    exit(0)
import random
import math
from stl import mesh
import numpy as np
import re

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
                                    help = "¿se está ejecutando con el python de Blender?")
        command_parser.add_argument('-b', '--blender',      type = str,  default = r"C:\Program Files\Blender Foundation\Blender 4.2",
                                    help = "ruta de la carpeta donde está blender.exe")
        command_parser.add_argument('-v', '--version',   type = str,  default = "4.2",  help = "versión del Blender") 
        command_parser.add_argument('-i', '--img',   type = str,    default = os.path.join("C:\\Users", os.getlogin(), "Desktop", "Img"),
                                    help = "ruta de la carpeta donde se guardarán las imagenes generadas")
        command_parser.add_argument('-n', '--num',   type = int,  default = 20, help = "número de ejemplos que se crean")  
        command_parser.add_argument('-s', '--stl',   type = str,  default = os.path.join("C:\\Users", os.getlogin(), "Desktop", "geom.stl"))
                                    
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
        
        # pip_path = os.path.join(args.blender, args.version, "python", "bin", "python.exe")
        # subprocess.run([pip_path, "-m", "pip", "install", "pandas"])
        
        blender_path = os.path.join(args.blender, args.version, "python", "bin", "python.exe")
        # Instalamos las librerias extra si no estan instaladas
        # subprocess.run([blender_path, "-m", "pip", "install", "parsy", "numpy-stl"])
        print(" ".join([blender_path, 
                        os.path.join(Path(os.path.dirname(__file__)).parent, "main.py"),
                        "prueba",
                        "--background",                   # Solo si se ejecuta desde blender
                        # # "-P",
                        # "--",                        
                        "-i",   args.img,
                        "-n",   str(args.num),
                        "-s",  args.stl
                        ]))
        subprocess.run([blender_path, 
                        os.path.join(Path(os.path.dirname(__file__)).parent, "main.py"),
                        "prueba",
                        "--background",                   # Solo si se ejecuta desde blender
                        # # "-P",
                        # "--",                        
                        "-i",   args.img,
                        "-n",   str(args.num),
                        "-s",  args.stl
                        ])    
        
        return None
    else:
        # Estamos trabajando desde el python de blender. Si no tenemos sus paquetes, hemos acabado
        
        blender_creation(args.img, args.stl, args.num)
            

def create_hexagonal_pyramid():
    bpy.ops.object.select_all(action='DESELECT')
    
    # Vértices para una base hexagonal centrada en el origen
    h = math.sqrt(3) / 2  # Altura del triángulo equilátero con lado 1
    # Lista de vertices
    verts = np.array([
        (1, 0, -1), (0.5, h, -1), (-0.5, h, -1),
        (-1, 0, -1), (-0.5, -h, -1), (0.5, -h, -1),
        (0, 0, 1)  # Vértice de la punta (altura de la pirámide)
    ])
        
    # Caras de la pirámide (6 triángulos y 1 hexágono en la base)
    faces = [
        (0, 1, 6), (1, 2, 6), (2, 3, 6), (3, 4, 6), (4, 5, 6), (5, 0, 6),
        (0, 1, 2, 3, 4, 5)  # Base
    ]
    
    # Crear la malla
    mesh_data = bpy.data.meshes.new("hex_pyramid_mesh")
    mesh_data_triangulated = bpy.data.meshes.new("hex_pyramid_mesh_triangulated")
    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()
    
    bm = bmesh.new()   # Crear un nuevo objeto bmesh
    bm.from_mesh(mesh_data)  # Cargar la malla existente en bmesh
    
    # Triangular las caras de la malla usando bmesh
    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    # Actualizar la malla con la nueva geometría triangulada
    bm.to_mesh(mesh_data_triangulated)
    bm.free()  # Liberar

    # Crear el objeto de la pirámide
    pyramid = bpy.data.objects.new("Hexagonal_Pyramid", mesh_data_triangulated)
    bpy.context.collection.objects.link(pyramid)
    bpy.context.view_layer.objects.active = pyramid
    pyramid.select_set(True)
    
    # Asegurarse de que esté centrado en el origen
    pyramid.location = (0, 0, 0)

    return pyramid  # Devolver el objeto de la pirámide para su posterior uso

# Función para rotar el objeto con una inclinación aleatoria
def convert_spherical():
    order = np.random.choice(['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX'])
    rotation_euler = Euler(np.random.random(3) * (2 * math.pi), order)
    rotation_matrix = rotation_euler.to_matrix()
    return np.array(rotation_matrix)

# Función para renderizar la imagen desde diferentes ángulos
def render_from_view(axis, rotation_matrix, geometry, image_path):
    # Configurar la cámara
    camera = bpy.data.objects.get("Camera")
    
    # Localizacion inicial de la camara
    start_time = time.time()
    camera.location = np.array((0, 0, 10) if axis == 'z' else (10, 0, 0) if axis == 'x' else (0, 10, 0))
    print("Tiempo el camera.location:", time.time() - start_time)

    # Rotamos la perspectiva de la camara para que mire al objeto
    if axis == 'x':
        track_to_simulator = Euler((math.pi/2, 0, math.pi/2), 'XYZ')
    elif axis == 'y':
        track_to_simulator = Euler((math.pi/2, 0, math.pi), 'XYZ')
    else:
        track_to_simulator = Euler((0,0,0), 'XYZ')

    camera.rotation_euler = track_to_simulator
    
    
    # Rotamos la camara
    camera.location = np.matmul(camera.location, np.array(rotation_matrix).T)
    # Rotamos en consecuencia la perspectiva de la camara
    camera.delta_rotation_euler = Matrix(rotation_matrix).to_euler('XYZ')
    
    # Configurar la cámara activa
    bpy.context.scene.camera = camera
    
    # CONFIGURACIÓN DE LA LUZ
    # Crear y configurar la luz
    light_data = bpy.data.lights.new(name="Light", type='POINT')
    light = bpy.data.objects.new(name="Light", object_data=light_data)
    bpy.context.collection.objects.link(light)
    
    # Colocar la luz en la posición de la cámara
    light.location = camera.location
    light.data.energy = 2000  # Ajusta la intensidad de la luz según sea necesario
    
    # Agregar el modificador 'Track To' para que la luz apunte al centro de la pirámide
    light_constraint = light.constraints.new(type='TRACK_TO')
    light_constraint.target = geometry # Usar la pirámide como objetivo
    light_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    light_constraint.up_axis = 'UP_Y'

    # Configurar la ruta de salida de la imagen
    bpy.context.scene.render.filepath = image_path
    
    # Renderizar
    bpy.ops.render.render(write_still=True)
    

# Crea un cierto numero de piramides en una cierta carpeta
def blender_creation(directory_gen, stl_path, num_samples):


    # Creación de directorios
    # directory_gen = os.path.join(output_folder, "Img")
    os.makedirs(directory_gen, exist_ok = True)
    directory_path_x = os.path.join(directory_gen, "Img_axis_x")
    directory_path_y = os.path.join(directory_gen, "Img_axis_y")
    directory_path_z = os.path.join(directory_gen, "Img_axis_z")
    os.makedirs(directory_path_x, exist_ok = True)
    os.makedirs(directory_path_y, exist_ok = True)
    os.makedirs(directory_path_z, exist_ok = True)

    # Eliminamos un escenario si lo hubiera
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    mesh_for_verts = mesh.Mesh.from_file(stl_path)
    verts = np.around(np.unique(mesh_for_verts.vectors.reshape([int(mesh_for_verts.vectors.size/3), 3]), axis=0),6)
    num_verts = verts.shape[0]
    print("Number of vertices are", num_verts)


    # Guardamos los resultados previos y añadimos los que tenemos
    if os.path.exists(os.path.join(directory_gen, "coords.csv")):
        matrix = np.loadtxt(os.path.join(directory_gen, "coords.csv"), delimiter=';')         # Matriz cuya primera linea es el ID y las otras tres son las coordenadas
        matrix = matrix.reshape((-1, 1 + num_verts*3))                                     # Asegurarse de que la matriz tiene la forma correcta cuando el csv solo contiene una iteracion         
        new_matrix = np.concatenate((matrix, np.zeros((num_samples, 1 + num_verts*3))), axis=0)
        previous_samples = matrix.shape[0]                                  # Número de ejemplos previos
        max_name = np.max(matrix[:,0])                                      # Número de ID máximo
    else:
        new_matrix = np.zeros((num_samples, 1 + num_verts*3), dtype=np.float32)
        previous_samples = 0                                                # Número de ejemplos previos
        max_name = 0                                                        # Número de ID máximo
    
    bpy.ops.wm.stl_import(filepath=stl_path)
    geometry_obj = bpy.data.objects[-1]
    bpy.context.view_layer.objects.active = geometry_obj
    geometry_obj.select_set(True)
    
    # Asegurarse de que esté centrado en el origen
    geometry_obj.location = (0, 0, 0)

    bpy.ops.object.camera_add()

    # Comenzamos a generar
    for j in range(num_samples):

        # Rotación aleatoria
        rotation_matrix = convert_spherical()

        # Obtener la posición del vértice superior después de la rotación
        verts_after_rotation = np.matmul(verts, rotation_matrix)

        # Generacion de los path
        num_name = int(j + max_name + 1)                                 # ID del sample
        for path_axis, axis in zip([directory_path_x, directory_path_y, directory_path_z],['x', 'y', 'z']):
            # Renderizar imágenes desde los ejes X, Y y Z y almacenar las rutas de las imágenes
            image_path = os.path.join(path_axis, f"sample_{num_name}_{axis}.png")
            # print(image_path)
            render_from_view(axis, rotation_matrix, geometry_obj, image_path)
        
        # Guardamos los datos
        new_matrix[previous_samples + j, 0] = num_name                      # ID de la imagen
        for i in range(num_verts):
            new_matrix[previous_samples + j, (1+i*3):((4+i*3))] = verts_after_rotation[i,:]        # Coordenadas
        
        
    # Guardar el dataframe en un archivo CSV
    csv_path = os.path.join(directory_gen, "coords.csv")  # Ruta para el CSV
    np.savetxt(csv_path, new_matrix, delimiter=';')