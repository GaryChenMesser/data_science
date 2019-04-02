import matplotlib.pyplot as plt
import numpy as np

with open("out1", 'r') as f:
    data = f.read().split('\n')

lam = eval(data[0])
print(len(lam))
print(len(data))
#print(data[1])
t_list = eval(data[0])
s_list = eval(data[2])

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
