from utils import iep_with_func
import numpy as np
import math
from copy import deepcopy

w = 0.0
w_update = 0.0
x = [1.0,0.64,0.36,0.3,0.2]
y = 1.0
alpha = 0.1
fct = [[['+',0]],[['+',1],['+',2]]]

def predict_new(w, x, y, alpha, level_fct):
    loss = 0.0
    loss1 = 0.0
    for fun in level_fct:
        copy = deepcopy(level_fct)
        copy.remove(fun)
        window_pred = np.dot(w, x[fun[1]])
        iep = iep_with_func(w,x,copy)
        loss += (iep + window_pred)
        print fun[1], x[fun[1]], iep, window_pred
    loss1 = iep_with_func(w,x,level_fct)
    return loss, loss1


def loss_new_scipy(w, x, y, alpha, level_fct):
    loss = 0.0
    for fun in level_fct:
    	copy = deepcopy(level_fct)
    	copy.remove(fun)
    	iep = iep_with_func(w,x,copy)
        window_pred = np.dot(w, x[fun[1]])
        print y, iep + window_pred
        loss += (y - iep - window_pred) ** 2
    return loss + alpha * math.sqrt(np.dot(w,w))


print fct
for epoch in range(5):
    for i_level,level_fct in enumerate(fct):
    	for fun in level_fct:
    		copy = deepcopy(level_fct)
    		copy.remove(fun)
    		w_update += (np.dot(w, x[fun[1]]) + iep_with_func(w,x,copy) - y) * (iep_with_func(1.0,x,copy) + x[fun[1]])
    w_update += 2 * w_update + 2 * alpha * w

    w -= 0.1 * w_update
    w_update = 0.0
    loss = 0.0

    for level_fct in fct:
        print loss
    	loss += loss_new_scipy(w, x, y, alpha, level_fct)
        print level_fct, loss
    print 'Loss', epoch, loss

for i_level,level_fct in enumerate(fct):
    ax = predict_new(w, x, y, alpha, level_fct)
    print i_level, ax