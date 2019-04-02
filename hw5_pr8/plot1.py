import matplotlib.pyplot as plt
import numpy as np

data = np.load("out1.npz")

lam = data['lam']
t_list = data['t_list']
s_list = data['s_list']

plt.figure()

for i in range(4):
    plt.subplot(221+i)
    plt.yscale('log')
    plt.plot(range(len(s_list[i])), s_list[i])
    plt.plot(range(len(t_list[i])), t_list[i])
    plt.legend(['primal residual', 'dual residual'])
    plt.xlabel('iteration')
    plt.title('lambda = {}'.format(lam[i]))

plt.show()
