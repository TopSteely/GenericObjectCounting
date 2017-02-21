from utils import iep_with_func
import numpy as np
import math
from copy import deepcopy
from scipy.optimize import minimize

w = 1.0
w_update = 0.0
x = [[1.0,0.64,0.36,0.3,0.2],[2.0,0.5,1.8,0.3]]
y = [1.0,2.0]
alpha = 0
fct = {0:[[['+',0]],[['+',1],['+',2]]], 1:[[['+',0]],[['+',1],['+',2],['-',3]]]}


def con(w,x):
    loss = 0.0
    for i_x in x:
        for i_i_x in i_x:
            loss += (np.dot(w, i_i_x))
    return loss


cons = ({'type': 'ineq', 'fun': con})#,
    #{'type': 'ineq', 'fun': con1})


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


def loss_new_scipy(w, x, y, alpha, fct):
    #print w
    loss = 0.0
    for img_nr, img_fct in zip(fct.keys(),fct.values()):
        print img_nr, img_fct
        for level_fct in img_fct:
            for fun in level_fct:
                copy = deepcopy(level_fct)
                copy.remove(fun)
                iep = iep_with_func(w,x[img_nr],copy)
                window_pred = np.dot(w, x[img_nr][fun[1]])
                #print y[img_nr], iep, window_pred
                if fun[0] == '+':
                    loss += ((y[img_nr] - iep - window_pred) ** 2)
                elif fun[0] == '-':
                    loss += ((y[img_nr] - iep + window_pred) ** 2)
                print level_fct, copy,iep, window_pred, y[img_nr], loss
    return loss + alpha * math.sqrt(np.dot(w,w))

print 'Loss', 'before', w, loss_new_scipy(w, x, y, alpha, fct)
print 'after'

#for epoch in range(8):
#    for img_nr, img_fct in zip(fct.keys(),fct.values()):
#        for level_fct in img_fct:
#            for fun in level_fct:
#                copy = deepcopy(level_fct)
#                copy.remove(fun)
#                w_update += (np.dot(w, x[img_nr][fun[1]]) + iep_with_func(w,x[img_nr],copy) - y[img_nr]) * (iep_with_func(1.0,x[img_nr],copy) + x[img_nr][fun[1]])
#    w_update += 2 * w_update + 2 * alpha * w

#    w -= 0.01 * w_update
#    w_update = 0.0
#    loss = 0.0

#    print 'Loss', epoch, w, loss_new_scipy(w, x, y, alpha, fct)
#res = minimize(loss_new_scipy, 1.0, args=(x, y, alpha, fct))
#print res
#for i_level,level_fct in enumerate(fct):
#    ax = predict_new(w, x, y, alpha, level_fct)
#    print i_level, ax