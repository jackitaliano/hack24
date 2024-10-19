import bpy
import os

def load_model():
    # Ensure the scene is clear (optional)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # get human model file path
    fbx_file_path = os.path.join(os.path.dirname(__file__), "obj", "scene.fbx")

    # Import the .obj file
    bpy.ops.import_scene.fbx(filepath=fbx_file_path)

    # Access the armature and mesh objects
    armature = bpy.data.objects["Armature"]  # Replace with the actual armature name
    mesh = bpy.data.objects["body_model_with_landmarks"]  # Replace with the actual mesh name

    # Print their scales to check if they match
    print(f"Armature scale: {armature.scale}")
    print(f"Mesh scale: {mesh.scale}")


def animate():
    pass

load_model()