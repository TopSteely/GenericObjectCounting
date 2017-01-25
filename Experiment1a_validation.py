import Input
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
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
	print len(data_to_scale)
	y.append(1)
for img_nr in negative_data:
	img_data = Data.Data(load_other, img_nr, 10, None)
	print len(data_to_scale)
	print len(img_data.X[randint(1,len(img_data.X))])
	data_to_scale.extend(img_data.X[randint(1,len(img_data.X))])
	data_to_scale.extend(img_data.X[randint(1,len(img_data.X))])
	y.append(0)
print len(data_to_scale)
scaler.fit(data_to_scale)

scaled = scaler.transform(data_to_scale)
sgd = SGDRegressor(eta0=math.pow(10,-5), learning_rate='invscaling', shuffle=True, average=True, alpha=0.00001)
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

for img_nr in other_test_d:
	img_data = Data.Data(load_other1, img_nr, 10, None)
	one = randint(1,len(img_data.X))

	sgd_error += sgd.predict(scaler.transform(img_data.X[one]))**2
	mlp1_error += mlp1.predict(scaler.transform(img_data.X[one]))**2
	mlp2_error += mlp2.predict(scaler.transform(img_data.X[one]))**2
	mlp3_error += mlp3.predict(scaler.transform(img_data.X[one]))**2
	mlp4_error += mlp4.predict(scaler.transform(img_data.X[one]))**2
	mlp5_error += mlp5.predict(scaler.transform(img_data.X[one]))**2

div_by = len(test_d) + len(other_test_d)
print 'SGD: ', sgd_error/div_by
print 'MLP1: ', mlp1_error/div_by
print 'MLP2: ', mlp2_error/div_by
print 'MLP3: ', mlp3_error/div_by
print 'MLP4: ', mlp4_error/div_by
print 'MLP5: ', mlp5_error/div_by
