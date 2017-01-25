import Input
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import math
import sys

class_ = sys.argv[1]

load_dennis = Input.Input('dennis',class_)
load_other = Input.Input('dennis','cat')
training_data = load_dennis.category_train
negative_data = load_other.category_train
scaler = StandardScaler()
data_to_scale = []
y = []
for img_nr in training_data:
	#if os.path.isfile('/var/node436/local/tstahl/GroundTruth/%s/%s.txt'%(class_,format(img_nr, "06d"))):
	#	gr = pd.read_csv('/var/node436/local/tstahl/GroundTruth/%s/%s.txt'%(class_,format(img_nr, "06d")), header=None, delimiter=",").values
	if os.path.isfile('/var/node436/local/tstahl/Features_groundtruth/Features_ground_truth/%s/%s.txt'%(class_,format(img_nr, "06d"))):
		feat = pd.read_csv('/var/node436/local/tstahl/Features_groundtruth/Features_ground_truth/%s/%s.txt'%(class_,format(img_nr, "06d")), header=None, delimiter=",").values
	print feat
	print feat.shape
	print feat.shape[0]
	raw_input()
	data_to_scale.extend(feat)
	y.append(1)
for img_nr in negative_data:
	img_data = Data.Data(load_dennis, img_nr, 10, None)
	one = np.random.rand(1,len(img_data.X))
	two = np.random.rand(1,len(img_data.X))
	data_to_scale.extend(img_data.X[one])
	data_to_scale.extend(img_data.X[two])
	y.append(0)
scaler.fit(data_to_scale)

scaled = scaler.transform(data_to_scale)
sgd = SGDRegressor(eta0=math.pow(10,-5), learning_rate='invscaling', shuffle=True, average=True, alpha=0.00001)
mlp1 = MLPRegressor(verbose=False, hidden_layer_sizes=(2000,500), alpha=al_i, activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
mlp2 = MLPRegressor(verbose=False, hidden_layer_sizes=(2000,1000), alpha=al_i, activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
mlp3 = MLPRegressor(verbose=False, hidden_layer_sizes=(1000,500), alpha=al_i, activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
mlp4 = MLPRegressor(verbose=False, hidden_layer_sizes=(2000,250), alpha=al_i, activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
mlp5 = MLPRegressor(verbose=False, hidden_layer_sizes=(500,500), alpha=al_i, activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
sgd.fit(scaled,y)
mlp1.fit(scaled,y)
mlp2.fit(scaled,y)
mlp3.fit(scaled,y)
mlp4.fit(scaled,y)
mlp5.fit(scaled,y)


sgd_error = 0.0
mlp1_error = 0.0
mlp2_error = 0.0
mlp3_error = 0.0
mlp4_error = 0.0
mlp5_error = 0.0
for img_nr in training_data:
	if os.path.isfile('/var/node436/local/tstahl/Features_groundtruth/Features_ground_truth/%s/%s.txt'%(class_,format(img_nr, "06d"))):
		feat = pd.read_csv('/var/node436/local/tstahl/Features_groundtruth/Features_ground_truth/%s/%s.txt'%(class_,format(img_nr, "06d")), header=None, delimiter=",").values
	sgd_error += sgd.predict(scaler.transform(feat))
	mlp1_error += mlp1.predict(scaler.transform(feat))
	mlp2_error += mlp2.predict(scaler.transform(feat))
	mlp3_error += mlp3.predict(scaler.transform(feat))
	mlp4_error += mlp4.predict(scaler.transform(feat))
	mlp5_error += mlp5.predict(scaler.transform(feat))