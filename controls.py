import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def read_landmark_map(map_id):
    filename = f"maps/landmarks_{map_id}.npy"
    return np.load(filename)


def generate_initial_pose(scene_width, scene_height):
    while True:
        x = np.random.uniform(0.2, scene_width - 0.2)
        y = np.random.uniform(0.2, scene_height - 0.2)
        theta = np.random.uniform(0, 2 * np.pi)
        if 0.2 <= x <= scene_width - 0.2 and 0.2 <= y <= scene_height - 0.2:
            return np.array([x, y, theta])


def generate_control_sequence(num_steps=200, step_duration=0.1, change_interval=2.0):
    num_changes = int(num_steps * step_duration / change_interval)
    controls = []
    for _ in range(num_changes):
        v = np.random.uniform(-0.5, 0.5)  # Linear velocity range
        w = np.random.uniform(-1.0, 1.0)  # Angular velocity range
        controls.extend([v, w] for _ in range(int(change_interval / step_duration)))
    return np.array(controls)


def is_valid_control_sequence(sequence, initial_pose, scene_width, scene_height):
    x, y, theta = initial_pose
    dt = 0.1  # Time step
    margin = 0.2  # Minimum distance from boundary

    for v, w in sequence:
        theta += w * dt
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt

        if not (margin <= x <= scene_width - margin and margin <= y <= scene_height - margin):
            return False

    return True


def save_control_sequence(initial_pose, control_sequence, map_id, seq_id):
    theta_placeholder = np.zeros((control_sequence.shape[0], 1))
    control_sequence_with_placeholder = np.hstack([control_sequence, theta_placeholder])

    # Now, stack the initial pose with the control sequence
    full_sequence = np.vstack([initial_pose, control_sequence_with_placeholder])

    # Save the sequence to a file
    if not os.path.exists('controls'):
        os.makedirs('controls')
    filename = f"controls/controls_{map_id}_{seq_id}.npy"
    np.save(filename, full_sequence)


def draw_robot(ax, pose, L=0.2, W=0.1):
    # Draws the robot as a rectangle on the plot
    x, y, theta = pose
    robot = patches.Rectangle((x - L/2, y - W/2), L, W, angle=np.rad2deg(theta), fill=True, color='red', ec='black')
    ax.add_patch(robot)


def plot_robot_path(landmarks, initial_pose, control_sequence, map_id, seq_id, scene_width, scene_height, draw_interval=10):
    fig, ax = plt.subplots()
    ax.plot(landmarks[:, 0], landmarks[:, 1], 'bo', markersize=5, label='Landmarks')
    x, y, theta = initial_pose
    dt = 0.1
    path = [initial_pose[:2]]

    for i, (v, w) in enumerate(control_sequence):
        theta += w * dt
        theta = (theta + np.pi) % (2 * np.pi) - np.pi  # Normalize theta
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        path.append([x, y])

        if i % draw_interval == 0:
            draw_robot(ax, (x, y, theta), L=0.2, W=0.1)  # Draw the robot at reduced intervals

    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], 'r-', label='Path')

    ax.set_xlim(0, scene_width)
    ax.set_ylim(0, scene_height)
    ax.set_title(f"Robot Path for Map {map_id}, Sequence {seq_id}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.legend()
    plt.savefig(f"controls/path_{map_id}_{seq_id}.png")
    plt.close()


def main():
    scene_width = 2.0
    scene_height = 2.0
    map_ids = [0, 1, 2, 3, 4]

    for map_id in map_ids:
        landmarks = read_landmark_map(map_id)
        for seq_id in [1, 2]:
            initial_pose = generate_initial_pose(scene_width, scene_height)
            control_sequence = None
            valid = False

            while not valid:
                control_sequence = generate_control_sequence()
                valid = is_valid_control_sequence(control_sequence, initial_pose, scene_width, scene_height)

            save_control_sequence(initial_pose, control_sequence, map_id, seq_id)
            plot_robot_path(landmarks, initial_pose, control_sequence, map_id, seq_id, scene_width, scene_height)


if __name__ == "__main__":
    main()
