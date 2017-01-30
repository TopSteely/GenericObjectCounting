import sys
import Input
import Output
import DummyData
import SGD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import math
import numpy as np
from sklearn.svm import SVC
import time
import random

epochs = 1

eta = 0.01

def main():
	sgd_blob = SGD.SGD('blob', 'mean', '', 5, 5, eta, 0.0001, 0.1, 3)
	scaler = StandardScaler()
	data_to_scale = []
	for im in sgd_blob.blobtraindata:
		data_to_scale.extend(im.X)
	scaler.fit(data_to_scale)

	#tested
	for im in sgd_blob.blobtraindata:
		im.scaler_transform(scaler)
	for im in sgd_blob.blobtestdata:
		im.scaler_transform(scaler)

	for ep in range(epochs):
		sgd_blob.learn()
	mse,ae, mse_non_zero = sgd_blob.evaluate('blobtest')
	mse_tr,ae_tr, mse_non_zero_tr = sgd_blob.evaluate('blobtrain')
	print "Eval loss train: ", mse_tr
	print "Eval loss val: ", mse

if __name__ == "__main__":
    main()