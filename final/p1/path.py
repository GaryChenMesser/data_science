import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy.stats import norm
import sys

def like(pro, epi, z, x):
    return z[0] * (z[1] * np.log(1 - epi[1]) + (1 - z[1]) * np.log(epi[1])) +\
           (1 - z[0]) * (z[1] * np.log(epi[0]) + (1 - z[1]) * np.log(1 - epi[0])) +\
           z[1] * (x * np.log(pro[1]) + (1 - x) * np.log(1 - pro[1])) +\
           (1 - z[1]) * (x * np.log(pro[0]) + (1 - x) * np.log(1 - pro[0]))

with open(sys.argv[1], 'r') as f:
    data = np.array(list(map(float, f.read().split('   '))), dtype = int) - 1


iteration = 1
n_cluster = 2
z         = np.zeros(data.shape, dtype = int)

print(data.shape)
print(z.shape)

for i in range(iteration):
    pro = [0.9, 0.5]
    epi = [0.1, 0.1]
    
    front = [data[0] * np.log(1 - pro[0]) + (1 - data[0]) * np.log(pro[0]), \
            data[0] * np.log(pro[1]) + (1 - data[0]) * np.log(1 - pro[1])]
    #accu = [0., 0.]
    #print(accu)
    for j in range(581):
        tmp = [0, 0.]
        for k in range(2):
            #print(k)
            #print(like(pro, epi, [0, k], data[j+1]) + accu[0])
            #print(like(pro, epi, [1, k], data[j+1]) + accu[1])
            for m in range(2):
                tmp[k] += np.exp(like(pro, epi, [m, k], data[j+1]) + front[m])
        
        front = np.log(tmp)
    print(front)
    print(np.exp(front))
    
    back = [data[581] * np.log(1 - pro[0]) + (1 - data[581]) * np.log(pro[0]), \
            data[581] * np.log(pro[1]) + (1 - data[581]) * np.log(1 - pro[1])]
    
    for j in range(581, data.shape[0] - 1):
        tmp = [0, 0.]
        for k in range(2):
            #print(k)
            #print(like(pro, epi, [0, k], data[j+1]) + accu[0])
            #print(like(pro, epi, [1, k], data[j+1]) + accu[1])
            for m in range(2):
                tmp[k] += np.exp(like(pro, epi, [m, k], data[j+1]) + back[m])
        
        back = np.log(tmp)
    
    accu = np.exp(like(pro, epi, [1, 0], data[581])) + np.exp(back[0])
    print(accu)
