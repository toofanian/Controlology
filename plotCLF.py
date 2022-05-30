from matplotlib import pyplot as plt
import torch
import numpy as np

clf = torch.load('controllers/trainedNetworks/singleint_60epoch_10penalty.pth')



x = np.arange(start=-2,stop=2,step=.001)
y = np.empty(x.shape)

for i in range(x.shape[0]):
    xi = torch.tensor(np.array([[x[i]]])).float()
    y[i] = clf(xi).detach().numpy()[0,0]

plt.plot(x,y)
plt.show()