import cv2
import cv2.aruco as aruco
import os

# Define marker types and their IDs
MARKER_TYPES = {
    0: "start",
    1: "goal",
    2: "turn_left_type_a",
    3: "turn_left_type_b",
    4: "turn_left_type_c",
    5: "turn_right",
    6: "lane_change_left",
    7: "lane_change_right",
    8: "go_forward",
    9: "end_of_turn",
    10: "intersection",
    11: "stop"
}

def generate_aruco_markers(dictionary_name, marker_types, marker_size=200, output_dir='aruco_markers'):
    os.makedirs(output_dir, exist_ok=True)
    aruco_dict = aruco.getPredefinedDictionary(dictionary_name)
    for marker_id, label in marker_types.items():
        img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        filename = f"{output_dir}/aruco_{marker_id}_{label}.png"
        cv2.imwrite(filename, img)
        print(f"Saved {filename} for {label}")

if __name__ == "__main__":
    DICT = aruco.DICT_4X4_50
    generate_aruco_markers(DICT, MARKER_TYPES)
    print("\nMarker ID mapping:")
    for marker_id, label in MARKER_TYPES.items():
        print(f"  {marker_id}: {label}")