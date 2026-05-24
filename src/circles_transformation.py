#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import make_circles

device = "mps"


# In[ ]:


fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

X_np, Y_np = make_circles(noise=0.1, factor=0.3, random_state=0)
# X_np, Y_np = make_moons(noise=0.1, random_state=0)

ax1.scatter(X_np[:, 0], X_np[:, 1], c=Y_np)
ax1.set_title("make_circles")

plt.tight_layout()
# plt.show()


# In[ ]:


hidden_units = 24

class CirclesModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(2, hidden_units)
        self.output = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x_hidden = self.hidden(x)
        return self.output(F.relu(x_hidden))


# In[ ]:


model = CirclesModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1)

epochs = 100
losses = []

X = torch.as_tensor(X_np, dtype=torch.float).to(device)
Y = torch.as_tensor(Y_np, dtype=torch.float).to(device)

model.train()
for epoch in range(epochs):
    optimizer.zero_grad()

    Y_pred = model(X)
    loss = criterion(Y_pred, Y.unsqueeze(dim=1))

    losses.append(loss.item())

    loss.backward()
    optimizer.step()

plt.plot(list(range(epochs)), losses)


# In[ ]:


model.eval()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1_range = np.linspace(-1.5, 1.5, 100)
x2_range = np.linspace(-1.5, 1.5, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)

batch_np = np.c_[X1.ravel(), X2.ravel()]
batch = torch.as_tensor(batch_np, dtype=torch.float).to(device)
with torch.no_grad():
    Z_flat = model(batch).numpy(force=True)
Z = Z_flat.reshape(X1.shape)
surf = ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none', alpha=0.7, zorder=1)
decision = ax.plot_surface(X1, X2, np.zeros_like(X1), zorder=1, alpha=0.5)

with torch.no_grad():
    Z_np = model(X).numpy(force=True).flatten()

A = np.where(Y_np == 0)
B = np.where(Y_np == 1)

x1 = X_np[A, 0]
x2 = X_np[A, 1]
ax.scatter(x1, x2, Z_np[A], marker='o', zorder=4)

x1 = X_np[B, 0]
x2 = X_np[B, 1]
ax.scatter(x1, x2, Z_np[B], marker='^', zorder=4)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()


# In[ ]:


params = [p.numpy(force=True) for p in model.parameters()]

hidden_weights = params[0]
hidden_biases = params[1]
output_weights = params[2]
output_bias = params[3]

Z_composite = []

for relu_idx in range(hidden_units):
    w1, w2 = hidden_weights[relu_idx]
    b = hidden_biases[relu_idx]
    o_w = output_weights[0][relu_idx]

    Z = o_w * np.maximum(0, (batch_np @ np.vstack([w1, w2])) + b)
    Z = Z.reshape(X1.shape)

    Z_composite.append(Z)

Z = np.sum(Z_composite, axis=0) + output_bias

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none', alpha=0.7, zorder=1)

plt.show()


# In[ ]:




