from utils import iep_with_func
import numpy as np
import math

w = 0.0
w_update = 0.0
x = [[1.0],[0.5],[0.5],[0.3],[0.2]]
y = 1.0
alpha = 0.1
fct = [[['+',0]],[['+',1],['+',2]]]

def predict_new(w, x, y, alpha, level_fct):
    loss = 0.0
    iep = iep_with_func(w,x,level_fct)
    print iep
    for fun in level_fct:
        window_pred = np.dot(w, x[fun[1]])
        loss += (iep - window_pred)
    return loss


def loss_new_scipy(w, x, y, alpha, level_fct):
    loss = 0.0
    iep = iep_with_func(w,x,level_fct)
    for fun in level_fct:
        window_pred = np.dot(w, x[fun[1]])
        loss += (y - iep - window_pred) ** 2
    return loss + alpha * math.sqrt(np.dot(w,w))


for epoch in range(5):
	level_preds = [np.dot(w, x[0])]
	level_preds.append(np.dot(w, x[1]) + np.dot(w, x[2]))
	for i_level,level_fct in enumerate(fct):
	    for fun in level_fct:
	        w_update += (np.dot(w, x[fun[1]]) + level_preds[i_level] - y) * (iep_with_func(w,x,level_fct) + x[fun[1]])
	w_update += 2 * w_update + 2 * alpha * w

	w -= 0.1 * w_update
	w_update = 0.0
	loss = 0.0
	for level_fct in fct:
		loss += loss_new_scipy(w, x, y, alpha, level_fct)
	print 'Loss', epoch, loss
	for level_fct in fct:
		print 'pred: ', predict_new(w, x, y, alpha, level_fct)