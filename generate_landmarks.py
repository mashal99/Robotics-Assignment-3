import numpy as np
import os


# Define a function to generate a single landmark map
def generate_landmark_map(num_landmarks, x_range, y_range):
    return np.random.uniform([x_range[0], y_range[0]], [x_range[1], y_range[1]], (num_landmarks, 2))


# Function to save the landmark map to a file
# Save the landmark map to a numpy file.
def save_landmark_map(landmark_map, map_id):

    if not os.path.exists('maps'):
        os.makedirs('maps')
    filename = f"maps/landmark_{map_id}.npy"
    np.save(filename, landmark_map)


# Generate and save landmark maps
num_maps = 5  # Total number of landmark maps
landmarks_per_map = [5, 5, 12, 12]  # Number of landmarks in each map

for map_id in range(num_maps):
    num_landmarks = landmarks_per_map[map_id % len(landmarks_per_map)]
    landmark_map = generate_landmark_map(num_landmarks, (0, 2), (0, 2))
    save_landmark_map(landmark_map, map_id)

# Return a message
print(f"Generated and saved {num_maps} landmark maps in the 'maps' directory.")

