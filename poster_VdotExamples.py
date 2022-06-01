
import math
from matplotlib import pyplot as plt
import torch
import numpy as np

x = np.arange(start=-3,stop=3,step=.01)
y = np.empty(x.shape)

def clf(x):
    return 2*x

for i in range(x.shape[0]):
    xi = torch.tensor(np.array([[x[i]]])).float()
    y[i] = clf(xi).detach().numpy()[0,0]

plt.plot(x,y,label=f'x^2')

def clf(x):
    return 4*x**3 + 2*x

for i in range(x.shape[0]):
    xi = torch.tensor(np.array([[x[i]]])).float()
    y[i] = clf(xi).detach().numpy()[0,0]

plt.plot(x,y,label=f'x^4+x^2')


def clf(x):
    return 2*x/(x**2+1)

for i in range(x.shape[0]):
    xi = torch.tensor(np.array([[x[i]]])).float()
    y[i] = clf(xi).detach().numpy()[0,0]

plt.plot(x,y,label=f'ln(x^2+1)')

#plt.fill_between(x, y,label='_nolegend_')

plt.tight_layout()
plt.xlim(left=-3,right=3)
plt.ylim(bottom=-10,top=10)

plt.legend(loc='best')
plt.grid(True)
plt.show()
