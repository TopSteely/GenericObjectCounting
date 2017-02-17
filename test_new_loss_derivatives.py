from utils import iep_with_func
import numpy as np
import math

w = 0.0
w_update = 0.0
x = [1.0,0.5,0.5,0.3,0.2]
y = 1.0
alpha = 0.1
fct = [[['+',0]],[['+',1],['+',2]]]

def predict_new(w, x, y, alpha, level_fct):
    loss = 0.0
    original_function = level_fct
    for fun in level_fct:
        copy = original_function
        copy.remove(fun)
        window_pred = np.dot(w, x[fun[1]])
        iep = iep_with_func(w,x,copy)
        loss += (iep - window_pred)
        print fun[1], x[fun[1]], iep, window_pred
    return loss


def loss_new_scipy(w, x, y, alpha, level_fct):
    loss = 0.0
    original_function = level_fct
    for fun in level_fct:
    	copy = original_function
    	copy.remove(fun)
    	iep = iep_with_func(w,x,copy)
        window_pred = np.dot(w, x[fun[1]])
        loss += (y - iep - window_pred) ** 2
    return loss + alpha * math.sqrt(np.dot(w,w))


for epoch in range(5):
	for i_level,level_fct in enumerate(fct):
		original_function = level_fct
		for fun in level_fct:
			copy = original_function
			copy.remove(fun)
			w_update += (np.dot(w, x[fun[1]]) + iep_with_func(w,x,copy) - y) * (iep_with_func(1.0,x,copy) + x[fun[1]])
	w_update += 2 * w_update + 2 * alpha * w

	w -= 0.01 * w_update
	w_update = 0.0
	loss = 0.0
	for level_fct in fct:
		loss += loss_new_scipy(w, x, y, alpha, level_fct)
	print 'Loss', epoch, loss

for i_level,level_fct in enumerate(fct):
    print predict_new(w, x, y, alpha, level_fct)