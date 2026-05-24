#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

device = "mps"


# In[ ]:


class PolyTorch(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc1(x)


# In[ ]:


# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

model = PolyTorch().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode="min", factor=0.8, patience=5, min_lr=1e-3
# )

func = lambda x: 2 * (x**2) + 5

X = np.linspace(-10, 10, 100) # might wanna shuffle this (np.random.permutation(N))
X_aug = np.stack([X**2, X], axis=-1)
y = func(X)

# X = torch.as_tensor(X, dtype=torch.float).to(device)
X_aug = torch.as_tensor(X_aug, dtype=torch.float).to(device)
y = torch.as_tensor(y, dtype=torch.float).to(device)

# normalization
X_mean = X_aug.mean(dim=0)
X_std = X_aug.std(dim=0)
X_aug = (X_aug - X_mean) / X_std

epochs = 1000
losses = []
model.train()
for _ in range(epochs):
    optimizer.zero_grad()

    pred_y = model(X_aug)

    loss = criterion(pred_y.squeeze(), y)
    losses.append(loss.item())

    # scheduler.step(loss.item())
    loss.backward()
    optimizer.step()

plt.plot(list(range(epochs)), losses)


# In[ ]:


# params = [p.numpy(force=True) for p in list(model.parameters())]
params = list(model.parameters())
print(params)


# In[ ]:


w = params[0] # (1, 2)
b = params[1] # (1,)

# y = Ax^2 + Bx + C

def forward_pass(X, w):
    # X: (100, 2)
    # w*: (2, 1)

    # X: (100, 2)
    # w: (2, N)
    return X @ w.T + b

def z_func(x_sqr, x):
    # x_sqr: (N,) use .ravel()
    # x: (N,)

    # treat w as though multi-neuron layer

    multineuron_w = np.stack([x_sqr, x], axis=-1)
    multineuron_w = torch.as_tensor(multineuron_w, dtype=torch.float).to(device)

    pred_y = forward_pass(X_aug, multineuron_w)
    y_expanded = torch.repeat_interleave(y.unsqueeze(dim=-1), x_sqr.shape[0], dim=-1)

    # output: (10000,)
    return torch.mean((pred_y - y_expanded)**2, dim=0).numpy(force=True)

w1_range = np.linspace(-50, 150, 100)
w2_range = np.linspace(-100, 100, 100)
X, Y = np.meshgrid(w1_range, w2_range)

zs = z_func(X.ravel(), Y.ravel())
Z = zs.reshape(X.shape)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

surface = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9, edgecolor="none")
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('Loss')


# In[ ]:


print(X_mean, X_std)


# In[ ]:




