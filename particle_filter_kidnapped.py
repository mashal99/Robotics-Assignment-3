import numpy as np
import matplotlib.pyplot as plt
import argparse

# Define noise standards for motion and sensor models
NOISE_STD = {'linear': 0.05, 'angular': 0.02}  # Example values for motion noise
SENSOR_NOISE_STD = {'distance': 0.1, 'angle': 0.05}  # Example values for sensor noise
DISTANCE_NOISE_STD = 0.1  # Example value for distance measurement noise standard deviation
ANGLE_NOISE_STD = 0.05    # Example value for angle measurement noise standard deviation

# Define environment bounds
x_min, x_max = 0.0, 10.0  # Example bounds for x-coordinate
y_min, y_max = 0.0, 10.0  # Example bounds for y-coordinate
theta_min, theta_max = -np.pi, np.pi  # Bounds for orientation (radians)
environment_bounds = [(x_min, x_max), (y_min, y_max), (theta_min, theta_max)]


# Parsing command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--map', type=str, required=True)
parser.add_argument('--sensing', type=str, required=True)
parser.add_argument('--num_particles', type=int, required=True)
parser.add_argument('--estimates', type=str, required=True)
args = parser.parse_args()


# Function to initialize particles uniformly
def initialize_particles_uniformly(num_particles, environment_bounds):
    x_min, x_max = environment_bounds[0]
    y_min, y_max = environment_bounds[1]
    theta_min, theta_max = environment_bounds[2]

    x_positions = np.random.uniform(x_min, x_max, num_particles)
    y_positions = np.random.uniform(y_min, y_max, num_particles)
    thetas = np.random.uniform(theta_min, theta_max, num_particles)

    particles = np.vstack((x_positions, y_positions, thetas)).T
    return particles


# Function to update particles based on motion model
def update_particles(particles, motion, noise_std):
    v, w = motion
    theta = particles[:, 2] + w + np.random.normal(0, noise_std[1], len(particles))
    particles[:, 0] += (v + np.random.normal(0, noise_std[0], len(particles))) * np.cos(theta)
    particles[:, 1] += (v + np.random.normal(0, noise_std[0], len(particles))) * np.sin(theta)
    particles[:, 2] = theta
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


# Main particle filter function
def particle_filter_kidnapped(map_file, sensing_file, num_particles, estimates_file):
    landmarks = np.load(map_file)
    sensor_data = np.load(sensing_file)

    # Define environment bounds (you need to set these according to your environment)
    environment_bounds = [(x_min, x_max), (y_min, y_max), (theta_min, theta_max)]

    # Initialize particles
    particles = initialize_particles_uniformly(num_particles, environment_bounds)

    # Placeholder for estimates
    estimates = []

    # Loop through sensor data to update particles and compute estimates
    for motion in sensor_data:
        particles = update_particles(particles, motion, NOISE_STD)
        weights = compute_weights(particles, motion, landmarks, SENSOR_NOISE_STD)
        particles = resample_particles(particles, weights)
        estimate = np.mean(particles, axis=0)  # Compute mean estimate
        estimates.append(estimate)

    # Save the estimated poses
    np.save(estimates_file, estimates)


# Run particle filter
if __name__ == "__main__":
    particle_filter_kidnapped(args.map, args.sensing, args.num_particles, args.estimates)
