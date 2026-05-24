#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

device = "mps"

hidden_units = 8


# In[ ]:


class PolyReluReg(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(1, hidden_units)
        self.output = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x_hidden = self.hidden(x)
        return self.output(F.relu(x_hidden))


# In[ ]:


func = lambda x: 2 * (x**2) + 5

criterion = nn.MSELoss()

model = PolyReluReg().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1)

X_numpy = np.linspace(-10, 10, 100)
y = func(X_numpy)

X = torch.as_tensor(X_numpy, dtype=torch.float).unsqueeze(dim=-1).to(device) # unsqueeze to add batch dim
y = torch.as_tensor(y, dtype=torch.float).to(device)

epochs = 100
losses = []
for _ in range(epochs):
    optimizer.zero_grad()

    pred_y = model(X)

    loss = criterion(pred_y.squeeze(), y)
    losses.append(loss.item())

    loss.backward()
    optimizer.step()

plt.plot(np.linspace(0, epochs, epochs), losses)
print(losses[-1])


# In[ ]:


params = [p.numpy(force=True) for p in model.parameters()]

hidden_weights = params[0]
hidden_biases = params[1]
output_weights = params[2]
output_bias = params[3]

def forward_pass_scalar(x):
    # 1. Transpose hidden_weights to align shapes: (1,) @ (1, 16) = (16,)
    hidden_layer_output = np.maximum(0, (x @ hidden_weights.T) + hidden_biases)

    # 2. Calculate the final scalar output
    # hidden_layer_output is (16,). output_weights must be (16,) to get a scalar.
    scalar_output = (hidden_layer_output @ output_weights.T) + output_bias

    return scalar_output


# In[ ]:


model.eval()

with torch.no_grad():
    plt.plot(X_numpy, model(X).numpy(force=True))
    # plt.plot(X_numpy, 2 * (X_numpy**2) + 5)

    Y_composite = []

    # decomposition
    for idx, w in enumerate(hidden_weights):
        h_w = w
        h_b = hidden_biases[idx]
        o_w = output_weights[0][idx]
        o_b = output_bias

        Y = np.maximum(0, X_numpy * h_w + h_b) * o_w
        Y_composite.append(Y)
        plt.plot(X_numpy, Y)

    Y_composite_arr = np.sum(Y_composite, axis=0) + o_b
    plt.plot(X_numpy, Y_composite_arr)


# In[ ]:




