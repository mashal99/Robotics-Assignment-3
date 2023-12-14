import numpy as np
import os

# Constants and noise parameters for odometry
DT = 0.1  # time step
WHEELBASE = 0.2  # wheelbase of the robot
ODOMETRY_NOISE_STD_LOW = {'linear': 0.05, 'angular': 0.1}
ODOMETRY_NOISE_STD_HIGH = {'linear': 0.1, 'angular': 0.3}


def add_noise(v, w, noise_std):
    # Add Gaussian noise to the velocities if they are not zero
    v_noisy = np.random.normal(v, noise_std['linear']) if v != 0 else 0
    w_noisy = np.random.normal(w, noise_std['angular']) if w != 0 else 0
    return v_noisy, w_noisy


def simulate_motion(pose, v, w, dt):
    """Update the robot's pose using the differential drive model."""
    x, y, theta = pose

    # Kinematic model for differential drive robot
    theta_new = theta + w * dt
    # Normalize theta to be within [-pi, pi)
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

    # Forward motion
    x_new = x + (v * np.cos(theta)) * dt
    y_new = y + (v * np.sin(theta)) * dt

    return np.array([x_new, y_new, theta_new])


def generate_odometry_readings(control_sequence, noise_std):
    """Generate noisy odometry readings for the given control sequence."""
    noisy_readings = []
    for v, w in control_sequence:
        # Apply noise to the control sequence
        v_noisy, w_noisy = add_noise(v, w, noise_std)
        noisy_readings.append([v_noisy, w_noisy])

    return np.array(noisy_readings)


def main():
    # Make sure required directories exist
    if not os.path.exists('readings'):
        os.makedirs('readings')

    # Loop over control sequence files
    for map_id in range(5):  # Assuming 5 maps
        for seq_id in [1, 2]:  # Two sequences per map
            # Load control sequence
            control_sequence = np.load(f'controls/controls_{map_id}_{seq_id}.npy')
            # Simulate motion and generate odometry readings for low noise
            odometry_readings_low = generate_odometry_readings(control_sequence, ODOMETRY_NOISE_STD_LOW)
            np.save(f'readings/readings_{map_id}_{seq_id}_L.npy', odometry_readings_low)
            # Simulate motion and generate odometry readings for high noise
            odometry_readings_high = generate_odometry_readings(control_sequence, ODOMETRY_NOISE_STD_HIGH)
            np.save(f'readings/readings_{map_id}_{seq_id}_H.npy', odometry_readings_high)


if __name__ == '__main__':
    main()
