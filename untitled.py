from utils import iep_with_func
import numpy as np
import math
from copy import deepcopy
from scipy.optimize import minimize

x = np.array([[[1.0,0.5],[0.64,0.32],[0.36,0.18],[0.3,0.15],[0.2,0.1]],[[0.1,0.05],[0.3,0.15],[0.6,0.3]]])
y = [[-1.0,-0.5,-0.4,-0.1,-0.2],[2.2,3.3,6.6]]
alpha = 0

def con(w,x,y,alpha):
    ret = 0.0
    for x_ in x:
    	print w, np.minimum(np.dot(w,x)_-1,0)
        ret += np.minimum(np.dot(w,x)_-1,0).sum()
    return ret

cons = ({'type': 'ineq', 'fun': con,'args':(x,y,alpha)})

def loss_new_scipy(w, x, y, alpha):
    loss = 0.0
    for x_,y_ in zip(x,y):
        for y_i,x_i in zip(y_,x_):
            loss += ((y_i - np.dot(w,x_i)) ** 2)
    return loss + alpha * math.sqrt(np.dot(w,w))

res = minimize(loss_new_scipy, np.array([1.0,2.0]), args=(x, y, alpha),constraints=cons,method='SLSQP')
print res
