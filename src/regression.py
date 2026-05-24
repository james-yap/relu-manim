#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

func = lambda x: x**2
noise = np.random.normal(0, 15, size=100)
# uniform_noise = np.random.uniform(-15, 15, size=100)

x = np.linspace(-10, 10, 100)
y = func(x)
y_noisy = func(x) + noise
# y_uni = func(x) + uniform_noise

coeff = np.polyfit(x, y_noisy, 2)
fit_func = np.poly1d(coeff)

plt.scatter(x, y_noisy)
# plt.scatter(x, y_uni)
plt.plot(x, fit_func(x))
plt.plot(x, y)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# 1. Define the true function
def my_func(x):
    return x**2

# 2. Generate x values
# 100 points evenly spaced between -10 and 10
x = np.linspace(-10, 10, 100)

# Calculate the true y values
y_true = my_func(x)

# 3. Add Gaussian noise
# np.random.normal(mean, standard_deviation, size)
noise_level = 15  # Increase this to make the data messier
noise = np.random.normal(0, noise_level, size=x.shape)
y_noisy = y_true + noise

# 4. Calculate lines of best fit using Least Squares (polyfit)
# Fit a straight line (Degree 1 polynomial: y = mx + b)
linear_coeffs = np.polyfit(x, y_noisy, 1)
linear_fit = np.poly1d(linear_coeffs)

# Fit a parabola (Degree 2 polynomial: y = ax^2 + bx + c)
quad_coeffs = np.polyfit(x, y_noisy, 2)
quad_fit = np.poly1d(quad_coeffs)

# 5. Plot everything
plt.figure(figsize=(10, 6))

# Plot the noisy data points
plt.scatter(x, y_noisy, color='gray', alpha=0.6, label='Noisy Data Points')

# Plot the original true function (hidden underneath the noise)
plt.plot(x, y_true, color='green', linestyle='--', linewidth=2, label='True Function ($y = x^2$)')

# Plot the lines of best fit
plt.plot(x, linear_fit(x), color='red', linewidth=2, label='Linear Fit (Underfit)')
plt.plot(x, quad_fit(x), color='blue', linewidth=2, label='Quadratic Fit (Best Fit)')

# Formatting the chart
plt.title('Visualizing Line of Best Fit with Noisy Data')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()

# Display the plot
plt.show()


# In[ ]:




