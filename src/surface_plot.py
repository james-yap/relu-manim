#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# 1. Define your mapping function (Z = f(X, Y))
def calculate_z(X, Y):
    # Example: A 3D ripple effect (sine wave based on distance from origin)
    R = np.sqrt(X**2 + Y**2)
    return np.sin(R)

# 2. Create 1D arrays for your X and Y bounds
# linspace(start, stop, number_of_points)
x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-5, 5, 100)

# 3. Create the 2D coordinate grids
# This is required for plot_surface to understand the grid layout
X, Y = np.meshgrid(x_vals, y_vals)

# 4. Calculate Z values for the entire grid
Z = calculate_z(X, Y)

# 5. Set up the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
# projection='3d' is the key to enabling 3D plots
ax = fig.add_subplot(111, projection='3d')

# 6. Plot the surface
# cmap applies a color gradient based on the Z value
# alpha controls transparency, edgecolor can add wireframe lines
surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

# 7. Add labels and a color bar
ax.set_title('3D Surface Plot Example')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# The colorbar helps interpret the Z values visually
fig.colorbar(surface, shrink=0.5, aspect=10, label='Z Value')

# 8. Render the plot
plt.show()


# In[ ]:




