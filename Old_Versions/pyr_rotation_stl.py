import os
import bpy
import random
import math
import pandas as pd
from mathutils import Vector
import re

# Función para crear una pirámide de base hexagonal centrada en el origen
def create_hexagonal_pyramid():
    bpy.ops.object.select_all(action='DESELECT')
    
    # Vértices para una base hexagonal centrada en el origen
    h = math.sqrt(3) / 2  # Altura del triángulo equilátero con lado 1
    verts = [
        (1, 0, -1), (0.5, h, -1), (-0.5, h, -1),
        (-1, 0, -1), (-0.5, -h, -1), (0.5, -h, -1),
        (0, 0, 1)  # Vértice de la punta (altura de la pirámide)
    ]
    
    # Caras de la pirámide (6 triángulos y 1 hexágono en la base)
    faces = [
        (0, 1, 6), (1, 2, 6), (2, 3, 6), (3, 4, 6), (4, 5, 6), (5, 0, 6),
        (0, 1, 2, 3, 4, 5)  # Base
    ]
    
    # Crear la malla
    mesh_data = bpy.data.meshes.new("hex_pyramid_mesh")
    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()

    # Crear el objeto de la pirámide
    pyramid = bpy.data.objects.new("Hexagonal_Pyramid", mesh_data)
    bpy.context.collection.objects.link(pyramid)
    bpy.context.view_layer.objects.active = pyramid
    pyramid.select_set(True)
    
    # Asegurarse de que esté centrado en el origen
    pyramid.location = (0, 0, 0)

    return pyramid  # Devolver el objeto de la pirámide para su posterior uso

# Función para rotar el objeto con una inclinación aleatoria

def rotate_randomly(obj):
    obj.rotation_euler[0] = random.uniform(0, math.pi * 2)  # Rotación en el eje X
    obj.rotation_euler[1] = random.uniform(0, math.pi * 2)  # Rotación en el eje Y
    obj.rotation_euler[2] = random.uniform(0, math.pi * 2)  # Rotación en el eje Z
    
def get_top_vertex_position(pyramid):
    
    # Actualizar la matriz del objeto para reflejar cualquier transformación pendiente
    bpy.context.view_layer.update()  # Esta línea es importante para que Blender actualice las matrices

    # El vértice superior en coordenadas locales (antes de la transformación)
    local_top_vertex = Vector((0, 0, 1))
    
    # Transformar la posición del vértice al espacio global
    global_top_vertex = pyramid.matrix_world @ local_top_vertex
    
    return global_top_vertex

# Función para renderizar la imagen desde diferentes ángulos
def render_from_view(axis, pyramid, image_path):
    # Configurar la cámara
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.location = (0, 0, 10) if axis == 'z' else (10, 0, 0) if axis == 'x' else (0, 10, 0)
    
    # Agregar el modificador 'Track To' para que apunte al centro de la pirámide
    camera_constraint = camera.constraints.new(type='TRACK_TO')
    camera_constraint.target = pyramid  # Usar la pirámide como objetivo
    camera_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    camera_constraint.up_axis = 'UP_Y'
    
    # Configurar la cámara activa
    bpy.context.scene.camera = camera
    
    # Crear y configurar la luz
    light_data = bpy.data.lights.new(name="Light", type='POINT')
    light = bpy.data.objects.new(name="Light", object_data=light_data)
    bpy.context.collection.objects.link(light)
    
    # Colocar la luz en la posición de la cámara
    light.location = camera.location
    light.data.energy = 1000  # Ajusta la intensidad de la luz según sea necesario
    
    # Agregar el modificador 'Track To' para que la luz apunte al centro de la pirámide
    light_constraint = light.constraints.new(type='TRACK_TO')
    light_constraint.target = pyramid  # Usar la pirámide como objetivo
    light_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    light_constraint.up_axis = 'UP_Y'

    # Configurar la ruta de salida de la imagen
    bpy.context.scene.render.filepath = image_path
    
    # Renderizar
    bpy.ops.render.render(write_still=True)
    
    # Eliminar la cámara y la luz después del render
    bpy.data.objects.remove(camera)
    bpy.data.objects.remove(light)

def find_highest_image_number(directory, axis):
    highest_number = -1
    image_pattern = re.compile(r'sample_(\d+)_\w+')
    
    for image_name in os.listdir(directory):
        match = image_pattern.match(image_name)
        if match:
            image_number = int(match.group(1))
            if image_number > highest_number:
                highest_number = image_number
    
    return highest_number

current_directory = os.getcwd()
directory_gen = os.path.join(current_directory, "Img")
os.makedirs(directory_gen, exist_ok = True)
directory_path_x = os.path.join(directory_gen, "Img_axis_x")
directory_path_y = os.path.join(directory_gen, "Img_axis_y")
directory_path_z = os.path.join(directory_gen, "Img_axis_z")
directory_path_stl = os.path.join(directory_gen, "STL")
os.makedirs(directory_path_x, exist_ok = True)
os.makedirs(directory_path_y, exist_ok = True)
os.makedirs(directory_path_z, exist_ok = True)
os.makedirs(directory_path_stl, exist_ok = True)

max_x = find_highest_image_number(directory_path_x, 'x')
max_y = find_highest_image_number(directory_path_y, 'y')
max_z = find_highest_image_number(directory_path_z, 'z')
max_stl = find_highest_image_number(directory_path_z, 'z')

max_n = max([max_x, max_y, max_z, max_stl])

if os.path.exists(os.path.join(directory_gen, "coords.csv")):
    df = pd.read_csv(os.path.join(directory_gen, "coords.csv"), sep=";")
else:
    df = pd.DataFrame(columns=['Img_sample', 'x', 'y', 'z'])

num_nuevos_casos = 250

for j in range(num_nuevos_casos):

    # Borrar objetos existentes en la escena
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Crear la pirámide centrada en el origen y obtener una referencia a ella
    pyramid = create_hexagonal_pyramid()

    # Rotarla aleatoriamente
    rotate_randomly(pyramid)

    # Actualizar la escena para asegurarse de que la rotación se aplica
    bpy.context.view_layer.update()

    # Obtener la posición del vértice superior después de la rotación
    top_vertex_position = get_top_vertex_position(pyramid)


    for axis in ['x', 'y', 'z']:
        # Renderizar imágenes desde los ejes X, Y y Z y almacenar las rutas de las imágenes
        if axis == 'x':
            image_path = os.path.join(directory_path_x, f"sample_{j + max_n + 1}_{axis}.png")
        if axis == 'y':
            image_path = os.path.join(directory_path_y, f"sample_{j + max_n + 1}_{axis}.png")
        if axis == 'z':
            image_path = os.path.join(directory_path_z, f"sample_{j + max_n + 1}_{axis}.png")
        render_from_view(axis, pyramid, image_path)
            
    # Obtener la posición del vértice superior después de la rotación
    top_vertex_position = get_top_vertex_position(pyramid)
    top_vertex_position=tuple(top_vertex_position)
    
    # Crear un dataframe de pandas con la posición del vértice superior y las rutas de las imágenes
    data = {
        'Img_sample': f"sample_{str(j+max_n+1)}",  # Convertir el vector a tupla
        'x': float(top_vertex_position[0]),
        'y': float(top_vertex_position[1]),
        'z': float(top_vertex_position[2])
    }
    df.loc[len(df)] = data
    
    # Generación del archivo stl
    stl_path = os.path.join(directory_path_stl, f"sample_{j + max_n + 1}.stl")
    bpy.context.view_layer.objects.active = pyramid
    pyramid.select_set(True)
    bpy.ops.export_mesh.stl(filepath=stl_path, use_selection=True)
    
# Guardar el dataframe en un archivo CSV
csv_path = os.path.join(directory_gen, "coords.csv")  # Ruta para el CSV
df.to_csv(csv_path, sep=";", index=False)