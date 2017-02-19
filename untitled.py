from utils import iep_with_func
import numpy as np
import math
from copy import deepcopy
from scipy.optimize import minimize

x = [1.0,0.64,0.36,0.3,0.2]
y = [1.0,0.5,0.4,-0.1,-0.2]
alpha = 0

def con(w,x):
    loss = 0.0
    for i_x in x:
        for i_i_x in i_x:
            loss += (np.dot(w, i_i_x))
    return loss


cons = ({'type': 'ineq', 'fun': con})

def loss_new_scipy(w, x, y, alpha):
    print w
    loss = 0.0
    for y_i,x_i in zip(y,x):
        loss += ((y_i - np.dot(w,x_i)) ** 2)
    return loss + alpha * math.sqrt(np.dot(w,w))

res = minimize(loss_new_scipy, 10.0, args=(x, y, alpha),constraints=cons)
print res