from utils import iep_with_func
import numpy as np
import math
from copy import deepcopy
from scipy.optimize import minimize

x = np.array([[[1.0],[0.64]],[[0.1]]])
y = [[10.0,1.28],[1]]
alpha = 0

def con(w,x,y,alpha):
    ret = 0.0
    for x_ in x:
    	print w,x_,np.dot(np.array(x_),w) ,np.minimum(np.dot(np.array(x_),w)-1,0)
        ret += np.minimum(np.dot(np.array(x_),w)-1,0).sum()
    return ret

def upper_constraint(w,x,y,alpha):
    ret = 0.0
    for x_,y_ in zip(x,y):
        print w, np.dot(np.array(x_),w),np.minimum(y_-np.dot(np.array(x_),w),0)
        ret += np.minimum(y_-np.dot(np.array(x_),w),0).sum()
    return ret

cons = ({'type': 'ineq', 'fun': upper_constraint,'args':(x,y,alpha)})

def loss_new_scipy(w, x, y, alpha):
    loss = 0.0
    for x_,y_ in zip(x,y):
        for y_i,x_i in zip(y_,x_):
            loss += ((y_i - np.dot(w,x_i)) ** 2)
    return loss + alpha * math.sqrt(np.dot(w,w))

res = minimize(loss_new_scipy, np.array([0.0]), args=(x, y, alpha),constraints=cons,method='SLSQP')
print res
