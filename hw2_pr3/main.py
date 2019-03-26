import time
import numpy as np
from scipy.linalg import cho_solve, cho_factor, qr , svd, diagsvd, inv

# ----Initialization----
n = 1100
p = [10, 1000]
Lambda = 0.1
repeat = 1

#beta_hat = []
elapse_time = np.zeros((2, 2, 3))

for _r in range(repeat):
    print(_r)
    X = [np.random.normal(0., 1., (n, _p)) for _p in p]
    beta = [np.ones((_p, 1)) for _p in p]
    W = [[np.zeros((_p, _p)), np.identity(_p) * np.sqrt(Lambda)] for _p in p]
    epislon = np.random.normal(0., 1., (n, 1))

    y = [(np.dot(X[i], beta[i]) + epislon) for i in range(2)]

    X_ridge = [[np.concatenate((X[i], -W[i][j])) for j in range(2)] for i in range(2)]
    Y_ridge = [np.concatenate((y[i], np.zeros((_p, 1)))) for i, _p in enumerate(p)]

    for i in range(2):
        for j in range(2):
            # cholesky
            start = time.time()
            XTX = np.dot(X_ridge[i][j].T, X_ridge[i][j])
            XTy = np.dot(X_ridge[i][j].T, Y_ridge[i])
            cho_solve(cho_factor(XTX), XTy)
            elapse_time[i][j][0] += time.time() - start
        
            # QR
            start = time.time()
            _, r = qr(X_ridge[i][j], mode = 'economic')
            XTy = np.dot(X_ridge[i][j].T, Y_ridge[i])
            cho_solve((r, False), XTy)
            elapse_time[i][j][1] += time.time() - start
        
            # SVD
            start = time.time()
            U, s, VH = svd(X_ridge[i][j])
            dim = X_ridge[i][j].shape
            SigIn = inv(diagsvd(s, dim[1], dim[1]))
            SigIn = np.concatenate((SigIn, np.zeros((dim[1], dim[0]-dim[1]))), axis = 1)
            np.dot(VH.T, np.dot(SigIn, np.dot(U.T, Y_ridge[i])))
            elapse_time[i][j][2] += time.time() - start
        
#print(beta_hat)
elapse_time /= repeat
print(elapse_time)
