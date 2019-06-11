import matplotlib.pyplot as plt
import numpy as np

with open("out1", 'r') as f:
    data = f.read().split('\n')
    alpha = eval(data[0])
    data = eval(data[1])

plt.figure()

for i in range(4):
    plt.subplot(221+i)
    plt.yscale('log')
    plt.plot(range(len(data[i][0])), data[i][0], 'k')
    print(len(data[i][0]))
    print(len(data[i][1]))
    plt.plot(range(len(data[i][1])), data[i][1], 'r')
    plt.legend(['proximal', 'fast proximal'])
    #plt.xlabel('iteration')
    plt.title(r'$\tilde{\alpha} = $' + str(alpha[i]))

plt.show()
#plt.savefig('plot1.png')
