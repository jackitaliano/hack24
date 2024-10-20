import bpy
import os

def load_model():
    # Ensure the scene is clear (optional)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # get human model file path
    fbx_file_path = os.path.join(os.path.dirname(__file__), "obj", "obj_and_armature.fbx")

    # Import the .obj file
    bpy.ops.import_scene.fbx(filepath=fbx_file_path, automatic_bone_orientation=True, force_connect_children=True)

    # Access the armature and mesh objects
    armature = bpy.data.objects["Armature"]  # Replace with the actual armature name
    mesh = bpy.data.objects["body_model_with_landmarks"]  # Replace with the actual mesh name

    # Print their scales to check if they match
    print(f"Armature scale: {armature.scale}")
    print(f"Mesh scale: {mesh.scale}")

    return armature, mesh


def animate():
    armature, mesh = load_model()

    if armature is None or mesh is None:
        print("Failed to load armature or mesh.")
        return

    # Switch to Pose Mode
    bpy.ops.object.mode_set(mode='POSE')

    # Clear existing animation context
    if armature.animation_data:
        armature.animation_data_clear()
    
    action = bpy.data.actions.new(name="MyAction")
    armature.animation_data_create()
    armature.animation_data.action = action

    # Keyframes definition
    keyframes = [
        (0, (0, 0, 0), (0, 0, 0, 0)),    # Frame 0: Original position and rotation
        (10, (0, 1, 0), (.1, 0, 0, 0)), # Frame 10: Move up and rotate
        (20, (0, 0, 0), (0, 0, 0, 0)),   # Frame 20: Return to original position
    ]

    # Move and rotate the bone, and insert keyframes
    bone_name = "Spine"  # Change to your actual bone name
    bone = armature.pose.bones.get(bone_name)

    if bone is None:
        print(f"Bone '{bone_name}' not found.")
        return

    for frame, position, rotation in keyframes:
        # Set the bone's location and rotation
        bone.location = position  # Change this to your desired position
        bone.rotation_quaternion = (rotation[0], rotation[1], rotation[2], rotation[3])  # Change this to your desired rotation
        # Insert keyframes for the location and rotation
        bone.keyframe_insert(data_path="location", frame=frame)
        bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)


    # Switch back to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Set the timeline frame range
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 20

# Call the animate function
animate()