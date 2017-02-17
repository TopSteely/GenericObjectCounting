from utils import iep_with_func
import numpy as np
import math
from copy import deepcopy
from scipy.optimize import minimize

x = [1.0,0.64,0.36,0.3,0.2]
y = [1.0,0.5,0.4,-0.1,-0.2]
alpha = 0

#cons = ({'type': 'ineq', 'fun': con1},
#    {'type': 'eq', 'fun': con2})

def loss_new_scipy(w, x, y, alpha):
    print w
    loss = 0.0
    for y_i,x_i in zip(y,x):
        loss += ((y_i - np.dot(w,x_i)) ** 2)
    return loss + alpha * math.sqrt(np.dot(w,w))

res = minimize(loss_new_scipy, 0.0, args=(x, y, alpha))
print res