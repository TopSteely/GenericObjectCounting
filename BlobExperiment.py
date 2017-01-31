import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
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

pr_mode = 'multi'

tree_level_size = 3

batch_size = 1

def main():
	for pr_mode in ['mean','multi']:
		for eta in [0.01]:
			sgd_blob = SGD.SGD('blob', pr_mode, '', tree_level_size, batch_size, eta, 0.0001, 0.1, 3)
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
				if pr_mode == 'multi':
					losses_imas_tr = np.array([], dtype=np.int64).reshape(tree_level_size,0)
					losses_imas_te = np.array([], dtype=np.int64).reshape(tree_level_size,0)
				else:
					losses_imas_tr = []
					losses_imas_te = []
				for imas in range(1,6):
					sgd_blob.reset_w()
					sgd_blob.learn('all',imas)
					mse,ae, mse_non_zero = sgd_blob.evaluate('blobtest')
					mse_tr,ae_tr, mse_non_zero_tr = sgd_blob.evaluate('blobtrain', imas)
					print "Eval loss train: ",eta, mse_tr
					print "Eval loss val: ",eta, mse
					if pr_mode == 'multi':
						#print losses_imas_tr.shape, mse_tr.reshape(-1,1).shape, mse_tr.shape
						losses_imas_tr = np.concatenate((losses_imas_tr,mse_tr.reshape(-1,1)), axis=1)#.reshape(-1,1)
						losses_imas_te = np.concatenate((losses_imas_te,mse.reshape(-1,1)), axis=1)#.reshape(-1,1)
					else:
						losses_imas_tr.append(mse_tr)
						losses_imas_te.append(mse)
			plt.plot(losses_imas_tr, '-rx', label='training error')
			plt.plot(losses_imas_te, '-go', label='test error')
			plt.legend()
			plt.xlabel('Training  samples')
			plt.ylabel('MSE')
			plt.legend('%s %s'%('blob',pr_mode))
			plt.savefig('/var/node436/local/tstahl/plos/blob_%s'%(pr_mode))

if __name__ == "__main__":
    main()