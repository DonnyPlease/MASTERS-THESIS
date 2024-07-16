import numpy as np
import matplotlib.pyplot as plt

# Constants
e = 1.602e-19  # Elementary charge in Coulombs
a0 = 5.29177e-11  # Bohr radius in meters
E_field = -1e10  # Negative electric field in V/m (example value)

# Potential function for hydrogen atom modified by electric field
def potential(x, E_field):
    V_hydrogen = -e**2 / (4 * np.pi * 8.854e-12 * np.abs(x))  # Coulomb potential
    V_field = e * E_field * x  # Potential energy due to electric field
    return V_hydrogen + V_field

x = np.linspace(0.01 * a0, 25 * a0, 1000)  # X-axis values (positive distance from nucleus)
V = potential(x, E_field)  # Calculate potential values
E_ground = -13.6  # Ground state energy of hydrogen atom (in eV)
V_eV = V / e  # Convert potential to eV for plotting

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x / a0, V_eV, label=r'$V_F(r) = V_H(r) - 10^{10}r$ V/m')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

# Highlighting the potential well
plt.fill_between(x / a0, V_eV, -15, where=(V_eV>=-15), color='red', alpha=0.2)

# plt.axhline(E_ground, color='green', linewidth=1.5, linestyle='--', label=f'Ground state energy = {E_ground} eV')

# Highlight the tunneling line at e_ground below the potential curve
tunnel_x = x[V_eV > E_ground]
tunnel_V = np.full_like(tunnel_x, E_ground)
plt.plot(tunnel_x / a0, tunnel_V, color='red', linewidth=2, linestyle='--', label='Tunneling region')

# Labels and title
plt.xlabel(r'$r$ [$a_B$]')
plt.ylabel(r'$V(r)$ [eV]')
plt.title("")
plt.legend()
plt.grid(True)
plt.ylim([-15, -5])
plt.xlim([0, 25])
plt.yticks(np.append(plt.yticks()[0], E_ground), np.append(plt.yticks()[1], f' {E_ground} eV'))
plt.ylim([-15, -5])

# Show plot
plt.show()
