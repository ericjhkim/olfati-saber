# Implementation of Olfati-Saber's paper: Flocking for Multi-Agent Dynamic Systems
#
# Created by Eric Kim (07-11-2024)

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D, art3d

# Constants
N_AGENTS = 10                               # Number of agents
SENSOR_RANGE = 20                           # Distance at which agents can sense each other
SIM_TIME = 10                               # Simulation time in seconds
dt = 0.1                                    # Simulation interval
DIMS = 3                                    # Number of dimensions
TGT_Q = [0,0,0]                             # Target position
TGT_P = [0,0,0]                             # Target velocity

# Directories
gif_path = "visualization.gif"

# Controls
CREATE_GIF = True
# np.random.seed(123)

# Initial conditions
def main():
    print("Starting simulation...")
    frame_count = 0

    swarm = Agents(N_AGENTS,SENSOR_RANGE,dt)
    alg = Algorithms(SENSOR_RANGE)
    obstacles = Obstacles(N_AGENTS,SENSOR_RANGE,dt)

    t = 0

    if CREATE_GIF:
        if not os.path.exists('frames'):
            os.makedirs('frames')

    with tqdm(total=SIM_TIME) as pbar:
        while t < SIM_TIME:
            A = swarm.get_adjacency()
            Ak = obstacles.get_adjacency(swarm)
            for i in range(N_AGENTS):
                u = 0
                q_i = swarm.states[:DIMS,i]
                p_i = swarm.states[DIMS:,i]
                for j in range(N_AGENTS):
                    if i != j and A[i,j] == 1: # Agent i is neighbour of agent j
                        q_j = swarm.states[:DIMS,j]
                        p_j = swarm.states[DIMS:,j]
                        u += alg.u_alpha(q_j,q_i,p_j,p_i)
                if DIMS == 3:
                    for k in range(obstacles.N_OBSTACLES):
                        if Ak[i,k] == 1: # Agent i is neighbour of obstacle k
                            q_k = obstacles.get_plane_position(q_i,obstacles.obs_plane_ak[k],obstacles.obs_plane_yk[k])
                            p_k = obstacles.get_plane_velocity(obstacles.obs_plane_ak[k],p_i)
                            u += alg.u_beta(q_k,q_i,p_k,p_i)
                u += alg.u_gamma(q_i,TGT_Q,p_i,TGT_P)
                swarm.update(u,i)

            t = round(t + dt, 10)
            pbar.update(min(dt, SIM_TIME - pbar.n))

            # Plot the swarm
            if CREATE_GIF:
                last = t+dt > SIM_TIME
                create_frame(swarm, A, t, frame_count, obstacles.plane_definitions, last)
                frame_count += 1

    # Create a GIF using PIL
    if CREATE_GIF:
        frames = []
        frame_files = [f'frames/frame_{i:04d}.png' for i in range(frame_count)]
        for frame_file in frame_files:
            frame = Image.open(frame_file)
            frames.append(frame)

        # Save the frames as a GIF
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=dt*1000, loop=0)

        # Clean up the frame images after the GIF is created
        import shutil
        shutil.rmtree('frames')

    plot_data(np.array(swarm.data))
    # animate_drone_positions(np.array(swarm.data), 'drone_positions.gif')

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

class Obstacles:
    def __init__(self, N_AGENTS, SENSOR_RANGE, dt):
        self.N_AGENTS = N_AGENTS
        self.r = SENSOR_RANGE
        self.dt = dt

        self.obs_plane_ak = []
        self.obs_plane_yk = []
        self.obstacle_type = "plane"

        self.plane_definitions = np.array([
            [[1, 0, 0], [0, 1, 0], [0, 0, 0]],          # Ground plane
            # [[0.1, 0, 1], [0, 1, 0], [10, 1, 1]]          # Vertical wall
        ])

        self.N_OBSTACLES = len(self.plane_definitions)

        if self.obstacle_type == "plane":
            self.build_planes()

    def get_adjacency(self, swarm):
        Ak = np.zeros((self.N_AGENTS,self.N_OBSTACLES))
        for i in range(self.N_AGENTS):
            for k in range(self.N_OBSTACLES):
                if np.linalg.norm(self.get_plane_position(swarm.states[:DIMS,i],self.obs_plane_ak[k],self.obs_plane_yk[k])) <= self.r:
                    Ak[i,k] = 1
        return Ak
    
    def build_planes(self):
        for plane in self.plane_definitions:
            v1 = plane[0]-plane[2]
            v2 = plane[1]-plane[2]
            y_k = plane[2]

            ak = np.cross(v1, v2)
            ak = ak/np.linalg.norm(ak)                      # Unit normal

            self.obs_plane_ak.append(ak)                    # Unit normal vector
            self.obs_plane_yk.append(y_k)                   # Point on plane
    
    # Get position of beta agent with respect to the obstacle
    def get_plane_position(self,q_i,a_k,y_k):
        P = np.identity(3)-np.outer(a_k,a_k.T)
        q_ik = np.matmul(P,q_i) + np.matmul(np.identity(3)-P,y_k)
        return q_ik-q_i

    def get_plane_velocity(self,a_k,p_i):
        P = np.identity(3)-np.outer(a_k,a_k.T)
        return np.matmul(P,p_i)
    
class Agents:
    def __init__(self, N_AGENTS, SENSOR_RANGE, dt):
        self.N_AGENTS = N_AGENTS
        self.r = SENSOR_RANGE
        self.dt = dt
        scaling_vector = np.concatenate([50*np.ones(DIMS),5*np.ones(DIMS)])
        self.states = np.random.uniform(-1,1,(2*DIMS, N_AGENTS)) * np.array(scaling_vector)[:, np.newaxis]
        if DIMS == 3: # state cannot be below ground
            self.states[2, :] = np.random.uniform(0.1, 1, N_AGENTS) * scaling_vector[2]
        self.data = [np.array(self.states)]

    def get_adjacency(self):
        A = np.zeros((self.N_AGENTS,self.N_AGENTS))
        for i in range(self.N_AGENTS):
            for j in range(self.N_AGENTS):
                if np.linalg.norm(self.states[:DIMS,i] - self.states[:DIMS,j]) <= self.r:
                    A[i,j] = 1
        return A

    def update(self,u,i_agent):
        self.states[DIMS:, i_agent] += u*self.dt           # Update velocity
        self.states[:DIMS, i_agent] += self.states[DIMS:, i_agent]*self.dt
        self.save_data()

    def save_data(self):
        self.data.append(np.array(self.states))

#%% Algorithms (implementing Eq 58)
class Algorithms:
    def __init__(self,SENSOR_RANGE):
        r = SENSOR_RANGE
        self.r = r                                      # Interaction range for agents
        self.r_prime = r                                # Interaction range for obstacle
        self.d = 0.75*r                                 # Desired distance between agents
        self.d_prime = 0.75*r                           # Desired distance between agent and obstacle

        self.c_alpha_1 = 1
        self.c_alpha_2 = 2*np.sqrt(self.c_alpha_1)
        self.c_beta_1 = 5
        self.c_beta_2 = 4
        self.c_gamma_1 = 5
        self.c_gamma_2 = 1

        self.a = 3                                      # Potential depth (attraction strength) 0 < a <= b
        self.b = 6                                      # Repulsion strength
        self.c = np.abs(self.a-self.b)/np.sqrt(4*self.a*self.b)

        self.h = 0.4                                    # Bump function parameter - interaction cutoff (smaller = sharper cutoff, larger = more fluid)
        self.epsilon = 0.1

    def u_alpha(self,q_j,q_i,p_j,p_i):
        u_i = self.c_alpha_1*self.phi_alpha(self.sigma_norm(q_j-q_i))*self.n_ij(q_j,q_i) + self.c_alpha_2*self.a_ij(q_j,q_i)*(p_j-p_i)
        return u_i

    def u_beta(self,q_k,q_i,p_k,p_i):
        u_i = self.c_beta_1*self.phi_beta(self.sigma_norm(q_k-q_i))*self.n_ij(q_k,q_i) + self.c_beta_2*self.b_ik(q_k,q_i)*(p_k-p_i)
        # print(self.phi_beta(self.sigma_norm(q_k-q_i))*self.n_ij(q_k,q_i),self.b_ik(q_k,q_i)*(p_k-p_i))
        return u_i
    
    # Navigational feedback term
    def u_gamma(self,q_i,q_r,p_i,p_r):
        u_i = -self.c_gamma_1*self.sigma_1(q_i-q_r) - self.c_gamma_2*(p_i-p_r)
        return u_i

    def sigma_1(self,z):
            return z/np.sqrt(1+z**2)

    # Action function (eq. 15)
    def phi_alpha(self,z):
        def phi(z):
            return 0.5*((self.a+self.b)*self.sigma_1(z+self.c)+(self.a-self.b))
        
        r_alpha = self.sigma_norm(self.r)
        d_alpha = self.sigma_norm(self.d)
        
        return self.rho_h(z/r_alpha) * phi(z-d_alpha)

    def phi_beta(self,z):
        d_beta = self.sigma_norm(self.d_prime)
        
        return self.rho_h(z/d_beta) * (self.sigma_1(z-d_beta)-1)

    # Bump function (eq. 10)
    def rho_h(self,z):
        if z >= 0 and z < self.h:
            return 1
        elif z >= self.h and z <= 1:
            return 0.5*(1+np.cos(np.pi*(z-self.h)/(1-self.h)))
        else:
            return 0
        
    # Sigma norm (eq. 8)
    def sigma_norm(self,z):
        return (np.sqrt(1+self.epsilon*np.linalg.norm(z)**2)-1)/self.epsilon

    # Gradient of the sigma norm (sigma_epsilon(z): eq. 9)
    def grad_sigma_norm(self,z):
        return z/np.sqrt(1+self.epsilon*np.linalg.norm(z)**2)

    def n_ij(self,q_j,q_i):
        return self.grad_sigma_norm(q_j-q_i)
    
    # Spatial adjacency matrix A(q)
    def a_ij(self,q_j,q_i):
        r_alpha = self.sigma_norm(self.r)
        return self.rho_h(self.sigma_norm(q_j-q_i)/r_alpha)

    def b_ik(self,q_k,q_i):
        d_beta = self.sigma_norm(self.d_prime)
        return self.rho_h(self.sigma_norm(q_k-q_i)/d_beta)

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

if __name__ == '__main__':
    main()

