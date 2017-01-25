import Input
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import math
import sys
import Data
from random import randint

def get_intersection_over_union(A, B):
    in_ = bool_rect_intersect(A, B)
    if not in_:
        return 0
    else:
        left = max(A[0], B[0]);
        top = max(A[1], B[1]);
        right = min(A[2], B[2]);
        bottom = min(A[3], B[3]);
        intersection = [left, top, right, bottom];
        surface_intersection = (intersection[2]-intersection[0])*(intersection[3]-intersection[1]);
        surface_A = (A[2]- A[0])*(A[3]-A[1]) + 0.0;
        surface_B = (B[2]- B[0])*(B[3]-B[1]) + 0.0;
        return surface_intersection / (surface_A + surface_B - surface_intersection)


class_ = sys.argv[1]

load_dennis = Input.Input('dennis',class_)
training_data = load_dennis.category_train
test_d = load_dennis.category_val

scaler = StandardScaler()
data_to_scale = []
y = []
for img_nr in training_data:
	if os.path.isfile('/var/node436/local/tstahl/GroundTruth/%s/%s.txt'%(class_,format(img_nr, "06d"))):
		gr = pd.read_csv('/var/node436/local/tstahl/GroundTruth/%s/%s.txt'%(class_,format(img_nr, "06d")), header=None, delimiter=",").values
	img_data = Data.Data(load_other, img_nr, 10, None)
	data_to_scale.extend(img_data.X)
	for bbox in img_data.boxes:
		count = 0.0
		for ground_truth in gr:
			count += get_intersection_over_union(bbox, ground_truth)
		y.append(count)
scaler.fit(data_to_scale)

scaled = scaler.transform(data_to_scale)
sgd  = SGDRegressor(eta0=math.pow(10,-5), learning_rate='invscaling', shuffle=True, average=True, alpha=0.00001)
mlp1 = MLPRegressor(verbose=False, hidden_layer_sizes=(2000,500), alpha=al_i, activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
mlp2 = MLPRegressor(verbose=False, hidden_layer_sizes=(2000,1000), alpha=al_i, activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
mlp3 = MLPRegressor(verbose=False, hidden_layer_sizes=(1000,500), alpha=al_i, activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
mlp4 = MLPRegressor(verbose=False, hidden_layer_sizes=(2000,250), alpha=al_i, activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
mlp5 = MLPRegressor(verbose=False, hidden_layer_sizes=(500,500), alpha=al_i, activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
print 'fitting'
sgd.fit(scaled,y)
mlp1.fit(scaled,y)
mlp2.fit(scaled,y)
mlp3.fit(scaled,y)
mlp4.fit(scaled,y)
mlp5.fit(scaled,y)

print 'fitted'
sgd_error = 0.0
mlp1_error = 0.0
mlp2_error = 0.0
mlp3_error = 0.0
mlp4_error = 0.0
mlp5_error = 0.0
for img_nr in test_d:
	if os.path.isfile('/var/node436/local/tstahl/Features_groundtruth/Features_ground_truth/%s/%s.txt'%(class_,format(img_nr, "06d"))):
		feat = pd.read_csv('/var/node436/local/tstahl/Features_groundtruth/Features_ground_truth/%s/%s.txt'%(class_,format(img_nr, "06d")), header=None, delimiter=",").values
	sgd_error += np.sum((sgd.predict(scaler.transform(feat)) - np.ones(feat.shape[0]))**2)
	mlp1_error += np.sum((mlp1.predict(scaler.transform(feat)) - np.ones(feat.shape[0]))**2)
	mlp2_error += np.sum((mlp2.predict(scaler.transform(feat)) - np.ones(feat.shape[0]))**2)
	mlp3_error += np.sum((mlp3.predict(scaler.transform(feat)) - np.ones(feat.shape[0]))**2)
	mlp4_error += np.sum((mlp4.predict(scaler.transform(feat)) - np.ones(feat.shape[0]))**2)
	mlp5_error += np.sum((mlp5.predict(scaler.transform(feat)) - np.ones(feat.shape[0]))**2)

div_by = len(test_d)
print 'SGD: ', sgd_error/div_by
print 'MLP1: ', mlp1_error/div_by
print 'MLP2: ', mlp2_error/div_by
print 'MLP3: ', mlp3_error/div_by
print 'MLP4: ', mlp4_error/div_by
print 'MLP5: ', mlp5_error/div_by
