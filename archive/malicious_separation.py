# Extension of Olfati-Saber's paper: Flocking for Multi-Agent Dynamic Systems
# This work adds a term to the control law to separate malicious agents from each other
#
# Created by Eric Kim (07-11-2024)

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import tools
from datetime import datetime

# Constants
N_AGENTS = 10                               # Number of agents
SENSOR_RANGE = 20                           # Distance at which agents can sense each other
SIM_TIME = 20                               # Simulation time in seconds
dt = 0.1                                    # Simulation interval
DIMS = 3                                    # Number of dimensions
TGT_Q = [0,0,5]                             # Target position
TGT_P = [0,0,0]                             # Target velocity

# Directories
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
gif_path = f"visualizations/anim_{TIMESTAMP}.gif"

# Controls
SAVE_DATA = True
CREATE_GIF = True
np.random.seed(123)

# Initial conditions
def main():
    print("Starting simulation...")
    frame_count = 0

    swarm = Agents(DIMS,N_AGENTS,SENSOR_RANGE,dt)
    alg = Algorithms(SENSOR_RANGE)
    obstacles = Obstacles(DIMS,N_AGENTS,SENSOR_RANGE,dt)

    t = 0

    if CREATE_GIF:
        if not os.path.exists('frames'):
            os.makedirs('frames')

    with tqdm(total=SIM_TIME) as pbar:
        while t < SIM_TIME:
            A = swarm.get_adjacency()
            # Ak = obstacles.get_adjacency(swarm)
            A_prime = swarm.get_sep_adjacency()
            for i in range(N_AGENTS):
                u = 0
                q_i = swarm.states[:DIMS,i]
                p_i = swarm.states[DIMS:,i]
                for j in range(N_AGENTS):
                    if i != j and A[i,j] > 0: # Agent i is neighbour of agent j
                        q_j = swarm.states[:DIMS,j]
                        p_j = swarm.states[DIMS:,j]
                        u += alg.u_alpha(q_j,q_i,p_j,p_i)
                    if i != j and A_prime[i,j] > 0: # Agent i is neighbour of agent j
                        # print(i,j,alg.u_sep(A_prime[i,j],q_j,q_i))
                        u += alg.u_sep(q_j,q_i)
                # if DIMS == 3:
                #     for k in range(obstacles.N_OBSTACLES):
                #         if Ak[i,k] == 1: # Agent i is neighbour of obstacle k
                #             q_k = obstacles.get_plane_position(q_i,obstacles.obs_plane_ak[k],obstacles.obs_plane_yk[k])
                #             p_k = obstacles.get_plane_velocity(obstacles.obs_plane_ak[k],p_i)
                #             u += alg.u_beta(q_k,q_i,p_k,p_i)
                u += alg.u_gamma(q_i,TGT_Q,p_i,TGT_P)
                swarm.update(u,i)

            t = round(t + dt, 10)
            pbar.update(min(dt, SIM_TIME - pbar.n))

            # Plot the swarm
            if CREATE_GIF:
                last = t+dt > SIM_TIME
                tools.create_frame(swarm, A, t, frame_count, obstacles.plane_definitions, last, swarm.env.gusts, F=swarm.F)
                frame_count += 1

    # Save data
    if SAVE_DATA:
        tools.save_to_h5py(np.array(swarm.data), filename=f"sim_data_{TIMESTAMP}", dataset_name="simulation")

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

    tools.plot_data(np.array(swarm.data))
    tools.plot_separation(np.array(swarm.data))
    # animate_drone_positions(np.array(swarm.data), 'drone_positions.gif')

class Obstacles:
    def __init__(self, DIMS, N_AGENTS, SENSOR_RANGE, dt):
        self.DIMS = DIMS
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
                if np.linalg.norm(self.get_plane_position(swarm.states[:self.DIMS,i],self.obs_plane_ak[k],self.obs_plane_yk[k])) <= self.r:
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
    
#%% Agents class (system dynamics)
class Agents:
    def __init__(self, DIMS, N_AGENTS, SENSOR_RANGE, dt):
        self.DIMS = DIMS
        self.N_AGENTS = N_AGENTS
        self.r = SENSOR_RANGE
        self.dt = dt
        scaling_vector = np.concatenate([50*np.ones(self.DIMS),5*np.ones(self.DIMS)])
        self.states = np.random.uniform(-1,1,(2*self.DIMS, N_AGENTS)) * np.array(scaling_vector)[:, np.newaxis]
        if self.DIMS == 3: # state cannot be below ground
            self.states[2, :] = np.random.uniform(0.1, 1, N_AGENTS) * scaling_vector[2]
        self.data = [np.array(self.states)]

        self.env = Environment(self.DIMS,self.N_AGENTS,self.dt)

        self.F = np.random.choice([0, 1], size=(N_AGENTS,N_AGENTS), p=[0.9, 0.1])
        self.F = np.zeros((N_AGENTS,N_AGENTS))
        self.F[2,3] = 1
        self.F[3,2] = 1
        self.F[3,4] = 1
        self.F[4,3] = 1

    def get_adjacency(self):
        A = np.zeros((self.N_AGENTS,self.N_AGENTS))
        for i in range(self.N_AGENTS):
            for j in range(self.N_AGENTS):
                if np.linalg.norm(self.states[:self.DIMS,i] - self.states[:self.DIMS,j]) <= self.r:
                    A[i,j] = 1
        return A
    
    def get_sep_adjacency(self):
        A = self.get_adjacency()
        A_prime = np.multiply(A, self.F)
        return A_prime

    def update(self,u,i_agent):
        u = self.env.generate_gust(u,i_agent)                   # Apply gusting
        self.states[self.DIMS:, i_agent] += u*self.dt           # Update velocity
        self.states[:self.DIMS, i_agent] += self.states[self.DIMS:, i_agent]*self.dt
        self.save_data()

    def save_data(self):
        self.data.append(np.array(self.states))

#%% Algorithms (implementing Eq 58)
class Algorithms:
    def __init__(self,SENSOR_RANGE):
        r = SENSOR_RANGE
        self.r = r                                      # Interaction range for agents
        self.r_prime = r                                # Interaction range for obstacle
        self.d = 0.5*r                                  # Desired distance between agents
        self.d_prime = 0.75*r                           # Desired distance between agent and obstacle

        self.c_alpha_1 = 1                              # Position-based attraction/repulsion between agents
        self.c_alpha_2 = 2*np.sqrt(self.c_alpha_1)      # Velocity-based alignment between agents
        self.c_beta_1 = 3                               # Position-based attraction/repulsion for obstacles
        self.c_beta_2 = 2*np.sqrt(self.c_beta_1)        # Velocity-based alignment for obstacles
        self.c_gamma_1 = 4                              # Navigational feedback term (position)
        self.c_gamma_2 = 1                              # Navigational feedback term (velocity)

        self.c_sep = 1                                  # Separation term

        self.a = 3                                      # Potential depth (attraction strength) 0 < a <= b
        self.b = 4                                      # Repulsion strength
        self.c = np.abs(self.a-self.b)/np.sqrt(4*self.a*self.b)

        self.h = 0.2                                    # Bump function parameter - interaction cutoff (smaller = sharper cutoff, larger = more fluid)
        self.epsilon = 0.1

    def u_sep(self,q_j,q_i):
        u_i = -self.c_sep * (q_i-q_j)/np.linalg.norm(q_i-q_j)
        return u_i

    def u_alpha(self,q_j,q_i,p_j,p_i):
        u_i = self.c_alpha_1*self.phi_alpha(self.sigma_norm(q_j-q_i))*self.n_ij(q_j,q_i) + self.c_alpha_2*self.a_ij(q_j,q_i)*(p_j-p_i)
        return u_i

    def u_beta(self,q_k,q_i,p_k,p_i):
        u_i = self.c_beta_1*self.phi_beta(self.sigma_norm(q_k-q_i))*self.n_ij(q_k,q_i) + self.c_beta_2*self.b_ik(q_k,q_i)*(p_k-p_i)
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

#%% Environmental factors
class Environment:
    def __init__(self,DIMS,N_AGENTS,dt):
        self.DIMS = DIMS
        self.N_AGENTS = N_AGENTS
        self.dt = dt
        self.gust_decay_rate = 5.0
        self.gust_probability = 0.0
        self.gusts = np.zeros((3,N_AGENTS))
        self.MAX_GUST_MAGNITUDE = 50

    def generate_gust(self,u,i_agent):
        if np.linalg.norm(self.gusts[:,i_agent]) <= 1e-2:
            self.gusts[:,i_agent] = np.zeros(self.DIMS)                     # Reset gust

            if np.random.uniform(0,1) < self.gust_probability:              # Generate gust
                magnitude = np.random.uniform(10,self.MAX_GUST_MAGNITUDE)
                direction = np.random.uniform(-1,1,self.DIMS)
                direction[2] *= 0.5                                         # Restrict gust in z direction
                direction /= np.linalg.norm(direction)
                self.gusts[:,i_agent] = magnitude*direction

        else:                                                               # Decay existing gust
            self.gusts[:,i_agent] *= np.exp(-self.gust_decay_rate*self.dt)
        
        u += self.gusts[:,i_agent]

        return u

if __name__ == '__main__':
    main()