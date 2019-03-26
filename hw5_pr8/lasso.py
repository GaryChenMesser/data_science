import numpy as np

n = 3000000
beta_true = np.array([-2, -2, 2, 2, -2])
X = np.zeros((n, 500), dtype=np.float32)
X = np.random.normal(0, 1, (n, 500))
y = np.matmul(X[:, ::100], beta_true) + np.random.normal(0, 0.05, (n))

Lambda = 0
rho = 1
m = 1000
row = n // m

theta = [np.zeros((m, 500)) for j in range(2)]
beta = [np.zeros((500,)) for i in range(2)]
alpha = [np.zeros((m, 500)) for j in range(2)]


inverse = []
XTy = []

print("test")

for i in range(m):
    inverse.append(np.linalg.inv(np.matmul(X[i * row:(i + 1) * row, :].T, X[i * row:(i + 1) * row, :]) + rho * np.identity(500)))
    XTy.append(np.matmul(X[i * row:(i + 1) * row, :].T, y[i * row:(i + 1) * row]))

print("test2")

for Lambda in [1]:
    t = 1
    s = 1
    t_stop = 0
    s_stop = 0
    
    while(t > t_stop or s > s_stop):
        theta[0] = theta[1]
        beta[0] = beta[1]
        alpha[0] = alpha[1]
        
        print(beta[0][::100])
        print(np.mean(theta[0][:, ::100], axis=0))
        
        for i in range(m):
            theta[1][i] = np.matmul(inverse[i], (XTy[i] + rho * (beta[0] - alpha[0][i])))
    
        _theta = np.mean(theta[1], axis=0)
        _alpha = np.mean(alpha[0], axis=0)
        beta[1] = np.sign(_theta + _alpha) * np.maximum(np.zeros(_theta.shape), np.abs(_theta + _alpha) - Lambda / rho / m)
    
        for i in range(m):
            alpha[1][i] = alpha[0][i] + theta[1][i] - beta[1]
            
        t = np.linalg.norm(np.mean([theta[1][i] - beta[1] for i in range(m)], axis=0))
        s = np.linalg.norm(beta[0] - beta[1])
        print(t)
        print(s)
        t_stop = 0.001 * np.linalg.norm(beta[1])
        s_stop = 0.001 * np.linalg.norm(np.mean(alpha[1], axis=0))
        
        if t > 10 * s:
            rho *= 2
            for i in range(m):
                alpha[1][i] /= 2
        elif s > 10 * t:
            rho /= 2
            for i in range(m):
                alpha[1][i] *= 2



