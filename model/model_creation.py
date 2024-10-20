import bpy
import os

# Ensure the scene is clear (optional)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# get human model file path
obj_file_path = os.path.join(os.path.dirname(__file__), "obj", "body_model_with_landmarks.obj")

# Import the .obj file
bpy.ops.wm.obj_import(filepath=obj_file_path)