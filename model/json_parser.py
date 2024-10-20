import json
import os

def json_transfomer():
    absolute_file_path = os.path.join(os.path.dirname(__file__), "data", "landmarks.json")

    with open(absolute_file_path, 'r') as file:
        data = json.load(file)

    # create a transformer to put it in the format needed for model_creation.py
    frames = []
    for map in data:
        # TODO: double check that names / conversions are correct
        current_dict = dict()
        current_dict['Left_Clavical'] = map['left_shoulder']
        current_dict['Right_Clavical'] = map['right_shoulder']
        current_dict['Left_Upper_Arm'] = map['left_elbow']
        current_dict['Right_Upper_Arm'] = map['right_elbow']
        current_dict['Left_Lower_Arm'] = map['left_wrist']
        current_dict['Right_Lower_Arm'] = map['right_wrist']
        current_dict['Left_Upper_Leg'] = map['left_knee']
        current_dict['Right_Upper_Arm'] = map['right_knee']
        current_dict['Left_Lower_Leg'] = map['left_ankle']
        current_dict['Right_Lower_Leg'] = map['right_ankle']
        frames.append(current_dict)
    
    return frames

print(json_transfomer())
