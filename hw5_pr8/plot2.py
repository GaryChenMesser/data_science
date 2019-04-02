import matplotlib.pyplot as plt
import numpy as np

with open("out2", 'r') as f:
    data = f.read()

lam = eval(data[0])
beta_list = eval(data[1])
covariate = [e for i, e in enumerate(beta_list) if (i+1)%100 == 0]
redundant = [e for i, e in enumerate(beta_list) if (i+1)%100 != 0]
print(len(covariate))
print(len(redundant))


plt.plot(range(len(t_list[i])), )
plt.legend(['primal residual', 'dual residual'])
plt.xlabel('iteration')
plt.title('lambda = {}'.format(lam[i]))

plt.show()
