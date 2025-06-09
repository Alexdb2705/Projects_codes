import bpy
import bmesh
import math
import addon_utils
import os

def decimate_stl(input_filepath, output_filepath, target_faces=49975):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    bpy.ops.wm.stl_import(filepath=input_filepath) # Imports of STL vary with the Blender version
    obj = bpy.context.selected_objects[0]

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    current_faces = len(bm.faces)
    bm.free()

    ratio = math.ceil(target_faces / current_faces * 10000) / 10000

    decimate_modifier = obj.modifiers.new(name='Decimate', type='DECIMATE')
    decimate_modifier.ratio = ratio

    bpy.ops.object.make_single_user(object=True, obdata=True, material=False, animation=False)

    bpy.ops.object.modifier_apply(modifier='Decimate') # Needs the extension "STL format (legacy)" to be installed manually on Blender

    bpy.ops.export_mesh.stl(filepath=output_filepath) # Exports of STL vary with the Blender version


folder_path = "C:\\Users\\Diego\\Desktop\\N101\\Datasets\\AeroSTL"
lista = [os.path.splitext(filename)[0] for filename in os.listdir(folder_path) if filename.endswith('.stl')]

lista=[
    "Avenger-716_UAV",
    "Grumman_F7F_Tigercat",
    "Scaneagle_UAV",
    "Drone_X8_quadrocopter_octocopter",
]

for geometry in lista:
    input_filepath = f"C:\\Users\\Diego\\Desktop\\N101\\Datasets\\AeroSTL\\{geometry}.stl"
    output_filepath = f"C:\\Users\\Diego\\Desktop\\N101\\Datasets\\STL_Blender\\{geometry}_50000.stl"

    decimate_stl(input_filepath, output_filepath)