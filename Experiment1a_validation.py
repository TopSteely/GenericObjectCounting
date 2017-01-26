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


class_ = sys.argv[1]

load_dennis = Input.Input('dennis',class_)
load_other = Input.Input('dennis','cat')
training_data = load_dennis.category_train
test_d = load_dennis.category_val

negative_data = load_other.category_train
load_other1 = Input.Input('dennis','bus')

other_test_d = load_other1.category_val
scaler = StandardScaler()
data_to_scale = []
y = []
for img_nr in training_data:
	#if os.path.isfile('/var/node436/local/tstahl/GroundTruth/%s/%s.txt'%(class_,format(img_nr, "06d"))):
	#	gr = pd.read_csv('/var/node436/local/tstahl/GroundTruth/%s/%s.txt'%(class_,format(img_nr, "06d")), header=None, delimiter=",").values
	if os.path.isfile('/var/node436/local/tstahl/Features_groundtruth/Features_ground_truth/%s/%s.txt'%(class_,format(img_nr, "06d"))):
		feat = pd.read_csv('/var/node436/local/tstahl/Features_groundtruth/Features_ground_truth/%s/%s.txt'%(class_,format(img_nr, "06d")), header=None, delimiter=",").values
	data_to_scale.extend(feat)
	for ff in range(feat.shape[0]):
		y.append(1)
for img_nr in negative_data:
	img_data = Data.Data(load_other, img_nr, 10, None)
	data_to_scale.append(img_data.X[randint(1,len(img_data.X)-1)])
	data_to_scale.append(img_data.X[randint(1,len(img_data.X)-1)])
	y.append(0)
	y.append(0)
print len(data_to_scale)
scaler.fit(data_to_scale)

scaled = scaler.transform(data_to_scale)
sgd = SGDRegressor(eta0=math.pow(10,-5), learning_rate='invscaling', shuffle=True, average=True, alpha=0.00001)
mlp1 = MLPRegressor(verbose=False, hidden_layer_sizes=(2000,500), activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
mlp2 = MLPRegressor(verbose=False, hidden_layer_sizes=(2000,1000), activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
mlp3 = MLPRegressor(verbose=False, hidden_layer_sizes=(1000,500), activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
mlp4 = MLPRegressor(verbose=False, hidden_layer_sizes=(2000,250), activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
mlp5 = MLPRegressor(verbose=False, hidden_layer_sizes=(500,500), activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
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
seen = 0
for img_nr in test_d:
	if os.path.isfile('/var/node436/local/tstahl/Features_groundtruth/Features_ground_truth/%s/%s.txt'%(class_,format(img_nr, "06d"))):
		feat = pd.read_csv('/var/node436/local/tstahl/Features_groundtruth/Features_ground_truth/%s/%s.txt'%(class_,format(img_nr, "06d")), header=None, delimiter=",").values
		for ff in range(feat.shape[0]): 
			correct_sgd += 1 if abs(sgd.predict(scaler.transform(feat[ff])) - 1) < 0.2 else 0
			correct_mlp1 += 1 if abs(mlp1.predict(scaler.transform(feat[ff])) - 1) < 0.2 else 0
			correct_mlp2 += 1 if abs(mlp2.predict(scaler.transform(feat[ff])) - 1) < 0.2 else 0
			correct_mlp3 += 1 if abs(mlp3.predict(scaler.transform(feat[ff])) - 1) < 0.2 else 0
			correct_mlp4 += 1 if abs(mlp4.predict(scaler.transform(feat[ff])) - 1) < 0.2 else 0
			correct_mlp5 += 1 if abs(mlp5.predict(scaler.transform(feat[ff])) - 1) < 0.2 else 0
			seen += 1

for img_nr in other_test_d:
	img_data = Data.Data(load_other1, img_nr, 10, None)
	one = randint(1,len(img_data.X)-1)

	correct_sgd += 1 if abs(sgd.predict(scaler.transform(img_data.X[one])) - 1) < 0.2 else 0
	correct_mlp1 += 1 if abs(mlp1.predict(scaler.transform(img_data.X[one])) - 1) < 0.2 else 0
	correct_mlp2 += 1 if abs(mlp2.predict(scaler.transform(img_data.X[one])) - 1) < 0.2 else 0
	correct_mlp3 += 1 if abs(mlp3.predict(scaler.transform(img_data.X[one])) - 1) < 0.2 else 0
	correct_mlp4 += 1 if abs(mlp4.predict(scaler.transform(img_data.X[one])) - 1) < 0.2 else 0
	correct_mlp5 += 1 if abs(mlp5.predict(scaler.transform(img_data.X[one])) - 1) < 0.2 else 0
	seen += 1

#div_by = len(test_d) + len(other_test_d)
print 'SGD: ', correct_sgd/seen
print 'MLP1: ', correct_mlp1/seen
print 'MLP2: ', correct_mlp2/seen
print 'MLP3: ', correct_mlp3/seen
print 'MLP4: ', correct_mlp4/seen
print 'MLP5: ', correct_mlp5/seen
