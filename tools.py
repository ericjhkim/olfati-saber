import os
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from datetime import datetime

#%% Plotting and Visualization
def plot_data(data):
    """
    Plots the position and velocity of drone agents from a 3D NumPy array.
    
    Parameters:
    - data: A 3D NumPy array of shape (N_TIME_STEPS, 4, N_AGENTS) where each 4xN_AGENTS sub-array
            represents the position (rows 0-2) and velocity (rows 2-4) of N_AGENTS drone agents.
    """
    num_time_steps = data.shape[0]
    num_drones = data.shape[2]  # Number of drone agents

    # Create subplots for position and velocity
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot positions
    for drone_id in range(num_drones):
        positions = data[:, 0:2, drone_id]  # Extract x and y positions over time
        axes[0].plot(positions[:, 0], positions[:, 1], label=f'Drone {drone_id + 1}')
    
    axes[0].set_title('Drone Positions Over Time')
    axes[0].set_xlabel('X Position')
    axes[0].set_ylabel('Y Position')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)
    
    # Plot velocities
    for drone_id in range(num_drones):
        velocities = data[:, 2:4, drone_id]  # Extract x and y velocities over time
        axes[1].plot(np.arange(num_time_steps), velocities[:, 0], label=f'Drone {drone_id + 1} X-Velocity')
        axes[1].plot(np.arange(num_time_steps), velocities[:, 1], linestyle='--', label=f'Drone {drone_id + 1} Y-Velocity')
    
    axes[1].set_title('Drone Velocities Over Time')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Velocity')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_separation(data):
    """
    Plots the average, minimum, and maximum separation distances among agents.

    Parameters:
        data (numpy.ndarray): A t x d x i array where:
                              t - number of timesteps,
                              d - number of dimensions (4 or 6),
                              i - number of agents.
    """
    # Ensure valid shape
    if data.ndim != 3:
        raise ValueError("Data must be a t x d x i array.")
    
    t, d, i = data.shape
    if d not in [4, 6]:
        raise ValueError("The dimension d must be either 4 or 6.")
    
    # Extract position data (first 2 or 3 dimensions)
    positions = data[:, :3 if d == 6 else 2, :]  # Shape: t x (2 or 3) x i
    
    # Compute pairwise distances
    distances = np.linalg.norm(
        positions[:, :, :, None] - positions[:, :, None, :], axis=1
    )  # Shape: t x i x i
    
    # Mask diagonal elements (self-distance)
    np.fill_diagonal(distances[0], np.nan)  # Mask diagonal on a sample timestep
    for step in range(1, t):
        np.fill_diagonal(distances[step], np.nan)
    
    # Compute statistics
    avg_separation = np.nanmean(distances, axis=(1, 2))
    min_separation = np.nanmin(distances, axis=(1, 2))
    max_separation = np.nanmax(distances, axis=(1, 2))
    
    # Plot results
    timesteps = np.arange(t)
    
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, avg_separation, label="Average Separation", color="blue")
    plt.fill_between(
        timesteps, min_separation, max_separation, color="lightblue", alpha=0.5,
        label="Range (Min to Max)"
    )
    plt.title("Separation Distance Analysis")
    plt.xlabel("Time Steps")
    plt.ylabel("Separation Distance")
    plt.legend()
    plt.grid()
    plt.show()

def create_frame(swarm, A, t, frame_count, planes, last):
    # Turn off interactive mode
    plt.ioff()

    def create_triangle_points_2d(x, y, vx, vy, scale=1.0):
        """Generate 2D triangle points oriented in the direction of the velocity vector with an additional 90-degree clockwise rotation."""
        if np.hypot(vx, vy) == 0:
            return np.array([[x, y]])  # Return a point if velocity is zero

        # Calculate angle of velocity vector
        angle = np.arctan2(vy, vx)

        # Define base triangle shape centered at the origin, pointing in the +x direction
        triangle_base = np.array([[scale, 0], [-scale, -scale * 0.5], [-scale, scale * 0.5]])

        # Create rotation matrix
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])

        # Rotate triangle points and translate to (x, y)
        rotated_triangle = (rotation_matrix @ triangle_base.T).T + np.array([x, y])
        return rotated_triangle

    def create_prism_points_3d(x, y, z, vx, vy, vz, scale=1.0):
        """Generate 3D triangular prism points oriented in the direction of the velocity vector."""
        direction_vector = np.array([vx, vy, vz])
        norm = np.linalg.norm(direction_vector)
        if norm == 0:
            return np.array([[x, y, z]])  # Return a single point if velocity is zero

        # Normalize the velocity vector
        direction_vector /= norm

        # Create triangle shape in the XZ plane with the nose pointing in the +x direction
        triangle_base = np.array([[scale, 0, 0], [-scale * 0.5, 0, scale * 0.5], [-scale * 0.5, 0, -scale * 0.5]])

        # Create rotation matrix to align x-axis with direction vector
        x_axis = np.array([1, 0, 0])
        v = np.cross(x_axis, direction_vector)
        s = np.linalg.norm(v)
        c = np.dot(x_axis, direction_vector)

        if s != 0:
            vx_matrix = np.array([[0, -v[2], v[1]],
                                [v[2], 0, -v[0]],
                                [-v[1], v[0], 0]])
            rotation_matrix = np.eye(3) + vx_matrix + (vx_matrix @ vx_matrix) * ((1 - c) / (s ** 2))
            rotated_triangle = (rotation_matrix @ triangle_base.T).T
        else:
            rotated_triangle = triangle_base

        return rotated_triangle + np.array([x, y, z])
    
    N_AGENTS = swarm.states.shape[1]

    if swarm.states.shape[0] == 6:
        # 3D plot
        fig = plt.figure(figsize=(5,4), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-50, 50])
        ax.set_ylim([-50, 50])
        ax.set_zlim([-10, 40])
        ax.set_title(f'Simulation Time: {t:.2f}s')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')

        for i in range(N_AGENTS):
            for j in range(N_AGENTS):
                if i != j and A[i, j] == 1:
                    x_values = [swarm.states[0, i], swarm.states[0, j]]
                    y_values = [swarm.states[1, i], swarm.states[1, j]]
                    z_values = [swarm.states[2, i], swarm.states[2, j]]
                    ax.plot(x_values, y_values, z_values, color='gray', linewidth=0.5, zorder=1)

        for i in range(N_AGENTS):
            x, y, z = swarm.states[0, i], swarm.states[1, i], swarm.states[2, i]
            vx, vy, vz = swarm.states[3, i], swarm.states[4, i], swarm.states[5, i]
            prism_points = create_prism_points_3d(x, y, z, vx, vy, vz, scale=1.5)
            poly = art3d.Poly3DCollection([prism_points], color='black')
            ax.add_collection3d(poly)

        ## Plotting obstacles
        # Plotting obstacles (planes)
        x = np.linspace(-50, 50, 50)
        y = np.linspace(-50, 50, 50)
        x, y = np.meshgrid(x, y)
        for plane in planes:
            v1 = np.array(plane[0])
            v2 = np.array(plane[1])
            point = np.array(plane[2])

            # Safeguard against division by zero by checking components
            if v1[0] != 0 and v2[1] != 0:
                z = point[2] + ((x - point[0]) * v1[2] / v1[0]) + ((y - point[1]) * v2[2] / v2[1])
                z = np.reshape(z, x.shape)  # Ensure z matches the grid's shape

                # Plot the surface with a specified color and transparency
                ax.plot_surface(x, y, z, alpha=0.5, color='magenta')

    else:
        # 2D plot
        fig, ax = plt.subplots(figsize=(5,4), dpi=300)
        ax.axis([-50, 50, -50, 50])
        ax.grid(True)
        ax.set_title(f'Simulation Time: {t:.2f}s')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

        for i in range(N_AGENTS):
            for j in range(N_AGENTS):
                if i != j and A[i, j] == 1:
                    x_values = [swarm.states[0, i], swarm.states[0, j]]
                    y_values = [swarm.states[1, i], swarm.states[1, j]]
                    ax.plot(x_values, y_values, color='gray', linewidth=0.5, zorder=1)

        for i in range(N_AGENTS):
            x, y = swarm.states[0, i], swarm.states[1, i]
            vx, vy = swarm.states[2, i], swarm.states[3, i]
            triangle_points = create_triangle_points_2d(x, y, vx, vy, scale=1.5)
            polygon = plt.Polygon(triangle_points, color='black')
            ax.add_patch(polygon)

    # Save the current frame as an image
    plt.savefig(f'frames/frame_{frame_count:04d}.png')
    if last:
        plt.show()
    plt.close(fig)  # Close the figure to prevent it from displaying

#%% Data Management
def save_to_h5py(data, filename="sim_data", dataset_name="data"):
    """
    Saves a numpy array to an HDF5 file using h5py, with a timestamp in the filename.

    Parameters:
        data (numpy.ndarray): The data array to save (t x d x i).
        filename (str): Base name of the file (default: 'sim_data').
        dataset_name (str): Name of the dataset inside the HDF5 file (default: 'data').
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    
    # Generate a timestamp
    filename = f"data\\{filename}.h5"

    with h5py.File(filename, "w") as h5file:
        h5file.create_dataset(dataset_name, data=data)
    
    print(f"Data saved to {filename} under dataset '{dataset_name}'.")

def load_latest_h5py(filename="sim_data", dataset_name="data", directory="data"):
    """
    Loads the most recent HDF5 file based on the timestamp in the filename.

    Parameters:
        filename (str): Base name of the file to look for (default: 'sim_data').
        dataset_name (str): Name of the dataset inside the HDF5 file (default: 'data').
        directory (str): Directory to search for the files (default: 'data').

    Returns:
        numpy.ndarray: The loaded data array from the latest file.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")
    
    # Compile a regex to match files with timestamps
    pattern = re.compile(f"{re.escape(filename)}_(\\d{{8}}_\\d{{6}})\\.h5$")
    
    # List all files in the specified directory
    files = os.listdir(directory)
    
    # Filter and extract valid timestamped files
    timestamped_files = []
    for file in files:
        match = pattern.match(file)
        if match:
            timestamped_files.append((file, match.group(1)))
    
    if not timestamped_files:
        raise FileNotFoundError(f"No files matching the pattern {filename}_<timestamp>.h5 found in '{directory}'.")
    
    # Sort files by timestamp
    timestamped_files.sort(key=lambda x: datetime.strptime(x[1], "%Y%m%d_%H%M%S"), reverse=True)
    latest_file = timestamped_files[0][0]
    
    # Load the latest file
    file_path = os.path.join(directory, latest_file)
    with h5py.File(file_path, "r") as h5file:
        data = h5file[dataset_name][:]
    
    print(f"Loaded data from {file_path} under dataset '{dataset_name}'.")
    return data