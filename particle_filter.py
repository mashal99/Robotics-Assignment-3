import numpy as np
import matplotlib.pyplot as plt
import argparse

# Parsing command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--map', type=str, required=True)
parser.add_argument('--sensing', type=str, required=True)
parser.add_argument('--num_particles', type=int, required=True)
parser.add_argument('--estimates', type=str, required=True)
args = parser.parse_args()


# Function to initialize particles
def initialize_particles(num_particles, initial_pose):
    # Create particles array with each particle having the initial_pose
    particles = np.tile(initial_pose, (num_particles, 1))
    return particles




# Function to update particles based on motion model
def update_particles(particles, motion):
    # Apply motion update (assuming motion = [v, w])
    v, w = motion
    theta = particles[:, 2] + w
    particles[:, 0] += v * np.cos(theta)  # Update x-coordinate
    particles[:, 1] += v * np.sin(theta)  # Update y-coordinate
    particles[:, 2] = theta  # Update orientation
    return particles


# Function to compute weights based on sensor readings
def compute_weights(particles, sensor_data, landmarks):
    weights = np.ones(len(particles))
    for i, particle in enumerate(particles):
        particle_x, particle_y, particle_theta = particle
        for landmark, (observed_distance, observed_angle) in zip(landmarks, sensor_data):
            # Predict distance and angle to the landmark from particle
            dx, dy = landmark[0] - particle_x, landmark[1] - particle_y
            predicted_distance = np.sqrt(dx**2 + dy**2)
            predicted_angle = np.arctan2(dy, dx) - particle_theta

            # Compute weight using Gaussian noise model
            distance_diff = observed_distance - predicted_distance
            angle_diff = observed_angle - predicted_angle
            weights[i] *= np.exp(-0.5 * (distance_diff**2 / DISTANCE_NOISE_STD**2 + angle_diff**2 / ANGLE_NOISE_STD**2))

    return weights / np.sum(weights)



# Function to resample particles
def resample_particles(particles, weights):
    indices = np.random.choice(range(len(particles)), size=len(particles), p=weights)
    resampled_particles = particles[indices]
    return resampled_particles



# Main function
def particle_filter(map_file, sensing_file, num_particles, estimates_file):
    landmarks = np.load(map_file)
    sensor_data = np.load(sensing_file)
    particles = initialize_particles(num_particles, sensor_data[0])  # Initial pose
    estimates = []

    for motion in sensor_data[1:]:
        particles = update_particles(particles, motion)
        weights = compute_weights(particles, motion, landmarks)
        particles = resample_particles(particles, weights)
        estimate = np.mean(particles, axis=0)  # Compute mean estimate
        estimates.append(estimate)

    np.save(estimates_file, estimates)

if __name__ == "__main__":
    particle_filter(args.map, args.sensing, args.num_particles, args.estimates)

