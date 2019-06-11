import matplotlib.pyplot as plt
import numpy as np

data = np.load('out2.npz')

lam = data['lam']
beta_list = data['beta_list']
covariate = beta_list[:, ::100]
redundant = np.delete(beta_list, np.s_[::100], 1)
print(covariate[0].shape)
print(redundant[0].shape)

print(redundant)

plt.plot(lam, redundant, '0.5')
plt.plot(lam, covariate, '0.0')

plt.savefig('plot2.png')
