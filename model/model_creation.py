import bpy
import os
import numpy as np

prevJointMap = \
{
'Left_Clavical': [    0.17295,    -0.30915,    -0.66739], 
'Right_Clavical': [   -0.23728,    -0.28415,      0.1915], 
'Left_Upper_Arm': [    0.33931,    -0.16232,     -1.0074], 
'Right_Upper_Arm': [   -0.22617,    -0.14275,     0.53093], 
'Left_Lower_Arm': [-0.00080627,     -0.1601,     -1.7631], 
'Right_Lower_Arm': [   -0.23589,   -0.021804,    -0.12531],
'Left_Upper_Leg': [    0.52694,     0.72389,    -0.50208],
'Right_Upper_Leg': [    0.31132,     0.72242,     0.54539],
'Left_Lower_Leg': [    0.82983,     0.88608,      0.3265],
'Right_Lower_Leg': [    0.39244,     0.91497,      1.4957]
}

currentJointMap = \
{
'Left_Clavical': [    0.17295,    -0.30915,    -0.66739], 
'Right_Clavical': [   -0.23728,    -0.28415,      0.1915], 
'Left_Upper_Arm': [    0.33931,    -0.16232,     -1.0074], 
'Right_Upper_Arm': [   -0.22617,    -0.14275,     0.53093], # this works
'Left_Lower_Arm': [-0.00080627,     -0.1601,     -1.7631], # this works
'Right_Lower_Arm': [   -0.23589,   -0.021804,    -0.12531],
'Left_Upper_Leg': [    0.52694,     0.72389,    -0.50208],
'Right_Upper_Leg': [    0.31132,     0.72242,     0.54539],
'Left_Lower_Leg': [    0.82983,     0.88608,      0.3265],
'Right_Lower_Leg': [    0.39244,     0.91497,      1.4957]
}

animation_frames=60

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

def set_to_rest():
    # Get the armature object
    armature_name = 'Armature'  
    armature = bpy.data.objects.get(armature_name)

    if armature and armature.type == 'ARMATURE':
        # Switch to Pose mode
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')
        
        # Clear all transformations
        for bone in armature.pose.bones:
            bone.location = (0, 0, 0)  # Reset location
            bone.rotation_euler = (0, 0, 0)  # Reset rotation (in Euler)
            bone.rotation_mode = 'XYZ'  # Set rotation mode to XYZ if needed
            bone.keyframe_insert(data_path="location")  # Insert keyframe for location
            bone.keyframe_insert(data_path="rotation_euler")  # Insert keyframe for rotation

        # Switch back to Object mode
        bpy.ops.object.mode_set(mode='OBJECT')

        print("Armature reset to rest state.")
    else:
        print("Armature not found or is not of type 'ARMATURE'.")

# helper method to calculate the rotations in each direction for a particular joint
def get_rotations(X_prev, X_next):
    D = (X_next[0] - X_prev[0], X_next[1] - X_prev[1], X_next[2] - X_prev[2])

    X_rot = np.arctan2(D[1], D[2])
    Y_rot = np.arctan2(D[0], np.sqrt(D[1]**2 + D[2]**2))
    Z_rot = np.arctan2(D[2], D[0])

    return (X_rot, Y_rot, Z_rot)

def animate():
    # set the model to rest before animating
    set_to_rest()

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

    for bone in armature.pose.bones:
        # calculate its rotation between prev position and current
        if bone.name in prevJointMap.keys() and bone.name in currentJointMap.keys():
            prevPos = prevJointMap[bone.name]
            currentPos = currentJointMap[bone.name]

            X_rot, Y_rot, Z_rot = get_rotations(prevPos, currentPos)

            keyframes = [
                (0, (0,0,0)),
                (animation_frames, (X_rot, Y_rot, Z_rot))
            ]

            for frame, rotation in keyframes:
                bone.rotation_mode = 'XYZ'
                bone.rotation_euler = rotation
                bone.keyframe_insert(data_path="rotation_euler", frame=frame)


    # Switch back to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Set the timeline frame range
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = animation_frames

# Call the animate function
animate()

