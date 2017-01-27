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

def bool_rect_intersect(A, B):
    return not (B[0]>A[2] or B[2]<A[0] or B[3]<A[1] or B[1]>A[3])

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
for img_nr in training_data[0:10]:
	if os.path.isfile('/var/node436/local/tstahl/GroundTruth/%s/%s.txt'%(class_,format(img_nr, "06d"))):
		gr = pd.read_csv('/var/node436/local/tstahl/GroundTruth/%s/%s.txt'%(class_,format(img_nr, "06d")), header=None, delimiter=",").values
	img_data = Data.Data(load_dennis, img_nr, 10, None)
	data_to_scale.extend(img_data.X)
	for bbox in img_data.boxes:
		count = 0.0
		for ground_truth in gr:
			count += get_intersection_over_union(bbox, ground_truth)
		y.append(count)
scaler.fit(data_to_scale)

scaled = scaler.transform(data_to_scale)
sgd1  = SGDRegressor(eta0=math.pow(10,-3), learning_rate='invscaling', shuffle=True, average=True)
sgd2  = SGDRegressor(eta0=math.pow(10,-4), learning_rate='invscaling', shuffle=True, average=True)
mlp1 = MLPRegressor(hidden_layer_sizes=(2000,500), activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
mlp2 = MLPRegressor(hidden_layer_sizes=(2000,1000), activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
mlp3 = MLPRegressor(hidden_layer_sizes=(1000,500), activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
mlp4 = MLPRegressor(hidden_layer_sizes=(2000,250), activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
mlp5 = MLPRegressor(hidden_layer_sizes=(500,500), activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
print 'fitting'
sgd1.fit(scaled,y)
sgd2.fit(scaled,y)
mlp1.fit(scaled,y)
mlp2.fit(scaled,y)
mlp3.fit(scaled,y)
mlp4.fit(scaled,y)
mlp5.fit(scaled,y)

print 'fitted'
sgd_error1 = 0.0
sgd_error2 = 0.0
mlp1_error = 0.0
mlp2_error = 0.0
mlp3_error = 0.0
mlp4_error = 0.0
mlp5_error = 0.0

sgd_preds1 = []
sgd_preds2 = []
mlp1_preds = []
mlp2_preds = []
mlp3_preds = []
mlp4_preds = []
mlp5_preds = []
y_p = []
for img_nr in test_d[0:10]:
	if os.path.isfile('/var/node436/local/tstahl/GroundTruth/%s/%s.txt'%(class_,format(img_nr, "06d"))):
			gr = pd.read_csv('/var/node436/local/tstahl/GroundTruth/%s/%s.txt'%(class_,format(img_nr, "06d")), header=None, delimiter=",").values
			img_data = Data.Data(load_dennis, img_nr, 10, None)
			for i_b,bbox in enumerate(img_data.boxes):
				count = 0.0
				for ground_truth in gr:
					count += get_intersection_over_union(bbox, ground_truth)

				sgd_error1 += (sgd1.predict(scaler.transform(img_data.X[i_b])) - count)**2
				sgd_preds1.append(sgd1.predict(scaler.transform(img_data.X[i_b])))
				sgd_error2 += (sgd2.predict(scaler.transform(img_data.X[i_b])) - count)**2
				sgd_preds2.append(sgd2.predict(scaler.transform(img_data.X[i_b])))
				mlp1_error += (mlp1.predict(scaler.transform(img_data.X[i_b])) - count)**2
				mlp1_preds.append(mlp1.predict(scaler.transform(img_data.X[i_b])))
				mlp2_error += (mlp2.predict(scaler.transform(img_data.X[i_b])) - count)**2
				mlp2_preds.append(mlp2.predict(scaler.transform(img_data.X[i_b])))
				mlp3_error += (mlp3.predict(scaler.transform(img_data.X[i_b])) - count)**2
				mlp3_preds.append(mlp3.predict(scaler.transform(img_data.X[i_b])))
				mlp4_error += (mlp4.predict(scaler.transform(img_data.X[i_b])) - count)**2
				mlp4_preds.append(mlp4.predict(scaler.transform(img_data.X[i_b])))
				mlp5_error += (mlp5.predict(scaler.transform(img_data.X[i_b])) - count)**2
				mlp5_preds.append(mlp5.predict(scaler.transform(img_data.X[i_b])))
				y_p.append(count)

div_by = len(test_d)
print div_by
print 'SGD1: ', sgd_error1/div_by
print 'SGD2: ', sgd_error2/div_by
print 'MLP1: ', mlp1_error/div_by
print 'MLP2: ', mlp2_error/div_by
print 'MLP3: ', mlp3_error/div_by
print 'MLP4: ', mlp4_error/div_by
print 'MLP5: ', mlp5_error/div_by

plt.figure()
for i_p, preds in enumerate([sgd_preds1,sgd_preds2,mlp1_preds,mlp2_preds,mlp3_preds,mlp4_preds,mlp5_preds]):
	sorted_preds = []
	decorated = [(y_i, i) for i, y_i in enumerate(y_p)]
	decorated.sort()
	for y_i, i in reversed(decorated):
	    sorted_preds.append(preds[i])
	    sorted_y.append(y_i)
	plt.plot(sorted_preds, 'ro',label='prediction')
	plt.plot(sorted_y, 'y*',label='target')
	plt.ylabel('y')
	plt.legend(loc='upper center')
	plt.savefig('/var/node436/local/tstahl/plos/Val1b_%s_%s.png'%(i_p,alpha,class_))
	plt.clf()