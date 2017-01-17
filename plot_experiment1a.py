import matplotlib
matplotlib.use('agg')
import pickle
import matplotlib.pyplot as plt
import math

classes = [
    #"person",
    "bird", "cat", "cow", "dog", "horse", "sheep",
    "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
    "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"
]

mse_path = "/home/tstahl/plot/1b_dennis_max_mse_%s_%s_%s.p"

for eta in [math.pow(10,-5),math.pow(10,-6)]:
	depths = []
	for tree_level_size in range(1,4):
		print tree_level_size
		error_tmp = 0.0
		for class_ in classes:
			print class_
			with open(mse_path%(class_,tree_level_size,eta), 'rb') as handle:
			    mse_tmp = pickle.load(handle)
			error_tmp += mse_tmp
		depths.append(error_tmp/len(classes))
	print eta
	if eta == math.pow(10,-5):
		print "if"
		plt.plot(depths, '-rx', label=eta)
	else:
		print "else"
		plt.plot(depths, '-bx', label=eta)
plt.legend()
plt.savefig("/var/node436/local/tstahl/plos/Experiment1a.png")
