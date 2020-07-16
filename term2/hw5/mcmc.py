# two things to do
# 1. diff
# 2. accept
import numpy as np
import sys
import matplotlib.pyplot as plt

with open(sys.argv[1], 'r') as f:
    pic = list(map(lambda x: x.split(' ')[:-1], f.read().split('\n')[:-2]))
    pic = (np.array(pic, dtype=int) + 1) / 2

ori_pic = pic.copy()

T = 1
rate = 0.4

def pi(pic, x, y, ori_pic):
    tmp = 0.
    if x > 0:
        tmp -= 1 - 2 * abs(pic[x][y] - pic[x-1][y])
    if x < pic.shape[0] - 1:
        tmp -= 1 - 2 * abs(pic[x][y] - pic[x+1][y])
    if y > 0:
        tmp -= 1 - 2 * abs(pic[x][y] - pic[x][y-1])
    if y < pic.shape[1] - 1:
        tmp -= 1 - 2 * abs(pic[x][y] - pic[x][y+1])
    
    #### diff
    
    #return abs(ori_pic[x][y] - pic[x][y]) * 0.6 / 0.4 + np.exp(1 / T * tmp)
    return np.exp(1 / T * tmp)

hop = 10000
choice = 150
diff = 0

for i in range(hop):
    print('\r{}'.format(i), end='')
    #sys.stdout.flush()
    choice_x = np.random.randint(pic.shape[0], size=choice)
    choice_y = np.random.randint(pic.shape[1], size=choice)
    best = 0.
    for c in range(choice):
        _pi = pi(pic, choice_x[c], choice_y[c], ori_pic)
        if best < _pi:
            best = _pi
            best_x, best_y = choice_x[c], choice_y[c]
    #if _pi < 1:
        #print(np.random.rand(1))
        #print(_pi)
    #    if np.random.rand() > _pi:
            #print('c')
    #        continue
    pic[best_x][best_y] = (pic[best_x][best_y] + 1) % 2
    
    
plt.imshow(ori_pic)
plt.show()
plt.close()
plt.imshow(pic)
plt.show()
    
