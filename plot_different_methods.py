import matplotlib
matplotlib.use('agg')
import pickle
import matplotlib.pyplot as plt
import math
import os

classes = [
    #"person",
    "cow",  "horse", "sheep","bird",#"cat","dog"
       "car", "motorbike", "train",#"aeroplane","bicycle","boat","bus",
    "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"
]

#self.mode, self.category, self.prune_tree_levels, eta0, alpha, learn_mode)
mse_path = "/home/tstahl/plot/1b_dennis_%s_mse_%s_%s_%s_%s_%s.p"

fig,ax = plt.subplots()
colors_ = ['r','g','b']

for eta in [math.pow(10,-5)]:
	for alpha in [math.pow(10,-5)]:
		for i_l,learn_mode in enumerate(['all']):#'category'
			for i_m,meth in enumerate(['max','mean']):#,'old'
				for tree_level_size in range(1,4):
					print tree_level_size
					depths = []
					error_tmp = 0.0
					for class_ in classes:
						print class_
						with open(mse_path%(meth,class_,tree_level_size,eta,alpha,learn_mode), 'rb') as handle:
						    mse_tmp = pickle.load(handle)
						error_tmp += mse_tmp
					depths.append(error_tmp/len(classes))
				ax[i_l].plot(depths, '-%so'%(colors_[i_m]), label='%s_%s'%(learn_mode,meth))
plt.legend()
plt.savefig("/var/node436/local/tstahl/plos/Experimentcomp.png")
