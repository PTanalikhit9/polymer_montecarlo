import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 10000  # Number of segments in the polymer chain
k = 2e-19  # Bending stiffness of the polymer, Nm²
delta_s = 25e-9 # Cellulose (monomer) size, m
L = delta_s * N
temp = 300  # Temperature, K
kB = 1.38e-23  # Boltzmann constant, J/K

# Initialize the polymer chain
theta = np.zeros(N-1)  # Initial angles between segments, assumed to be 0

def bending_energy(k, delta_s, theta):
    return (k / 2) * np.sum(1 - np.cos(theta)) * delta_s

def metropolis(E_old, E_new, T):
    if E_new < E_old:
        return True  # Accept if the new configuration has lower energy
    else:
        p = np.exp(-(E_new - E_old) / (kB * T))  # Metropolis criterion
        return np.random.rand() < p

# Monte Carlo Simulation
n_steps = 10000  # Number of Monte Carlo steps
for step in range(n_steps):
    # Randomly select a segment and propose a new angle
    i = np.random.randint(N-1)
    delta_theta = np.random.normal(0, 0.1)  # Small random change
    theta_new = theta.copy()
    theta_new[i] += delta_theta
    
    # Calculate energies and decide whether to accept the new configuration
    E_old = bending_energy(k, delta_s, theta)
    E_new = bending_energy(k, delta_s, theta_new)
    
    if metropolis(E_old, E_new, temp):
        theta = theta_new  # Accept the new configuration

# Calculate Final Bending Energy
final_bending_energy = bending_energy(k, delta_s, theta)

# Random Distribution of Monomers (Gaussian)
mean_ratio = 0.5  # example: 50% gelatin on average, adjust as needed
std_dev_ratio = 0.1  # example: standard deviation of 10%, adjust as needed
while True:  # Ensure valid probabilities are generated
    p_gelatin = np.random.normal(mean_ratio, std_dev_ratio)
    if 0 <= p_gelatin <= 1:
        break
p_cellulose = 1 - p_gelatin

# Randomly assign segment types according to the generated probabilities
segment_types = np.random.choice(['gelatin', 'cellulose'], size=N, p=[p_gelatin, p_cellulose])

# Count monomers of each species
count_gelatin = np.sum(segment_types == 'gelatin')
count_cellulose = np.sum(segment_types == 'cellulose')

colors = {'gelatin': 'b', 'cellulose': 'g'}  # blue for gelatin, green for cellulose

# Plot the resulting polymer chain with legend showing counts
x, y = [0], [0]
for i in range(N-1):
    x_end = x[-1] + np.cos(np.sum(theta[:i+1])) * delta_s
    y_end = y[-1] + np.sin(np.sum(theta[:i+1])) * delta_s
    plt.plot([x[-1], x_end], [y[-1], y_end], color=colors[segment_types[i]])
    x.append(x_end)
    y.append(y_end)

# Showing Legend with Counts and Proper Colors
legend_gelatin = plt.Line2D([0], [0], color='b', lw=2)
legend_cellulose = plt.Line2D([0], [0], color='g', lw=2)
plt.legend([legend_gelatin, legend_cellulose], [f'Gelatin: {count_gelatin}', f'Cellulose: {count_cellulose}'], loc='upper right')

plt.axis('equal')
plt.title(f'Polymer Chain using Monte Carlo Simulation\nFinal Bending Energy: {final_bending_energy:.2e} J')
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.grid(False)  # Ensure grid is off
plt.show()





