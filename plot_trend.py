import matplotlib
matplotlib.use('agg')
import pickle
import matplotlib.pyplot as plt
import math
import os

classes = [
    #"person",
    "bird", "cat", "cow", "dog", "horse", "sheep",
    "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
    "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"
]

mse_path = "/home/tstahl/plot/1b_dennis_max_mse_%s_%s_%s.p"

descending2 = []
descending3 = []

for class_ in classes:
	for eta in [math.pow(10,-5),math.pow(10,-6)]:
		for alpha in [math.pow(10,0),math.pow(10,-1),math.pow(10,-2),math.pow(10,-3),math.pow(10,-4),math.pow(10,-5),math.pow(10,-6),0]:
			for m_mode in ['max','mean']:
				if os.path.isfile('/home/tstahl/plot/1b_dennis_%s_mse_%s_%s_%s_%s_category.p'%(m_mode,class_,1,eta, alpha)):
					previous = 1000
					for tree_level_size in range(1,4):
						if os.path.isfile('/home/tstahl/plot/1b_dennis_%s_mse_%s_%s_%s_%s_category.p'%(m_mode,class_,tree_level_size,eta, alpha)):
							temp_sting = '%s_%s_%s_%s'%(class_,m_mode,eta,alpha)
							#print temp_sting, tree_level_size
							with open('/home/tstahl/plot/1b_dennis_%s_mse_%s_%s_%s_%s_category.p'%(m_mode,class_,tree_level_size,eta,alpha), 'rb') as handle:
								mse_tmp = pickle.load(handle)
							if tree_level_size == 1:
								baseline = mse_tmp
							print mse_tmp, previous
							if mse_tmp < previous:
								if tree_level_size == 2:
									descending2.append(temp_sting)
								elif tree_level_size == 3 and previous < baseline:
									descending3.append(temp_sting)
							previous = mse_tmp
print descending2
print descending3

exit()
for eta in [math.pow(10,-5),math.pow(10,-6)]:
	for alpha in [math.pow(10,0),math.pow(10,-1),math.pow(10,-2),math.pow(10,-3),math.pow(10,-4),math.pow(10,-5),math.pow(10,-6)]:

		for tree_level_size in range(1,4):
			print tree_level_size
			depths = []
			error_tmp = 0.0
			for class_ in classes:
				print class_
				with open(mse_path%(class_,tree_level_size,eta), 'rb') as handle:
				    mse_tmp = pickle.load(handle)
				error_tmp += mse_tmp
			depths.append(error_tmp/len(classes))
		print eta
		if eta == math.pow(10,-5):
			plt.plot(depths, '-rx', label=eta)
		else:
			plt.plot(depths, '-bx', label=eta)
plt.legend()
plt.savefig("/var/node436/local/tstahl/plos/Experiment1a.png")
