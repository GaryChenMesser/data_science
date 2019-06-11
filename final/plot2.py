import matplotlib.pyplot as plt
import numpy as np

with open("zero_fpg.out", 'r') as f:
    data1 = f.read().split('\n')
    alpha = eval(data1[0])
    data1 = eval(data1[1])
with open("zero_rpbcd.out", 'r') as f:
    data2 = f.read().split('\n')
    data2 = eval(data2[1])

plt.figure()

for i in range(2):
    plt.subplot(121+i)
    plt.yscale('log')
    plt.plot(range(len(data1[i])), data1[i], 'k')
    plt.plot(range(len(data2[i])), data2[i], 'r')
    plt.legend(['fpg', 'rpbcd'])
    #plt.xlabel('iteration')
    plt.title(r'$\lambda^\ast = $' + str(alpha[i]))

#plt.show()
plt.savefig('plot2.png')
