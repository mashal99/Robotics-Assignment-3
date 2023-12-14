import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse


def load_data(file_path):
    return np.load(file_path)


def dead_reckon(pose, v, w, dt):
    x, y, theta = pose
    theta += w * dt
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    return np.array([x, y, theta])


def animate(i, ground_truth, estimated_path, ax):
    ax.clear()
    ax.plot(ground_truth[:i, 0], ground_truth[:i, 1], 'g-')  # Ground truth path
    ax.plot(estimated_path[:i, 0], estimated_path[:i, 1], 'r-')  # Estimated path
    # Add more plotting details like landmarks here


def main(map_file, execution_file, sensing_file):
    landmarks = load_data(map_file)
    ground_truth = load_data(execution_file)
    odometry = load_data(sensing_file)

    estimated_path = [ground_truth[0]]
    for v, w in odometry:
        estimated_path.append(dead_reckon(estimated_path[-1], v, w, controlsDT))
    estimated_path = np.array(estimated_path)

    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, animate, frames=len(ground_truth),
                                  fargs=(ground_truth, estimated_path, ax), interval=100)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, required=True)
    parser.add_argument('--execution', type=str, required=True)
    parser.add_argument('--sensing', type=str, required=True)
    args = parser.parse_args()
    main(args.map, args.execution, args.sensing)
