import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


def load_data(file_path):
    return np.load(file_path)


def calculate_errors(ground_truth, estimates):
    translational_errors = []
    rotational_errors = []

    for gt, est in zip(ground_truth, estimates):
        # Translational error: Euclidean distance between the positions
        translational_error = np.linalg.norm(gt[:2] - est[:2])
        translational_errors.append(translational_error)

        # Rotational error: Smallest difference in angles considering circular topology
        rotational_error = np.arctan2(np.sin(gt[2] - est[2]), np.cos(gt[2] - est[2]))
        rotational_errors.append(rotational_error)

    return np.array(translational_errors), np.array(rotational_errors)


def update_animation(i, ground_truth, estimates, ground_truth_scatter, estimates_scatter):
    # Update the positions of the ground truth and estimates scatter plots
    ground_truth_scatter.set_offsets(ground_truth[i, :2])
    estimates_scatter.set_offsets(estimates[i, :2])

    return ground_truth_scatter, estimates_scatter


def create_animation(ground_truth, estimates, landmarks):
    fig, ax = plt.subplots()
    ax.plot(landmarks[:, 0], landmarks[:, 1], 'bo', label='Landmarks')

    ground_truth_line, = ax.plot([], [], 'b-', label='Ground Truth')
    estimates_line, = ax.plot([], [], 'k--', label='Estimate')
    ax.legend()

    def init():
        ground_truth_line.set_data([], [])
        estimates_line.set_data([], [])
        return ground_truth_line, estimates_line

    def animate(i):
        ground_truth_line.set_data(ground_truth[:i+1, 0], ground_truth[:i+1, 1])
        estimates_line.set_data(estimates[:i+1, 0], estimates[:i+1, 1])
        return ground_truth_line, estimates_line

    ani = FuncAnimation(fig, animate, frames=len(ground_truth), init_func=init, blit=True, interval=100)
    plt.show()
    return ani


def plot_errors(errors):
    # Assuming 'errors' is a tuple or list containing two arrays: (translational_errors, rotational_errors)
    translational_errors, rotational_errors = errors

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot translational errors
    axs[0].plot(translational_errors, label='Translational Error')
    axs[0].set_title('Translational Error over Time')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Error')

    # Plot rotational errors
    axs[1].plot(rotational_errors, label='Rotational Error')
    axs[1].set_title('Rotational Error over Time')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Error')

    # Show the plot
    plt.tight_layout()
    plt.legend()
    plt.show()


def main(map_file, execution_file, estimates_file):
    ground_truth = load_data(execution_file)
    estimates = load_data(estimates_file)
    landmarks = load_data(map_file)

    errors = calculate_errors(ground_truth, estimates)
    plot_errors(errors)

    ani = create_animation(ground_truth, estimates, landmarks)

    # Save the animation
    ani.save('robot_animation.mp4', writer='ffmpeg')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, required=True)
    parser.add_argument('--execution', type=str, required=True)
    parser.add_argument('--estimates', type=str, required=True)
    args = parser.parse_args()

    main(args.map, args.execution, args.estimates)