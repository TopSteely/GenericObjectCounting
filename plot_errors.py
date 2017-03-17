import matplotlib
matplotlib.use('agg')
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np

methods = ['counting','full_image','gt','grid','avg','avg_level']#trancos,mscoco,full image,'gt','grid','counting','avg','avg_level'


scores = {}
for method in methods:
    with open('/var/scratch/spintea/Repositories/ms-caffe/output/visualization/error_per_level_%s.pickle'%(method),'rb') as handle:
        error_per_level = pickle.load(handle)
        if len(error_per_level) < 9:
        	scores[method] = np.concatenate((error_per_level,np.zeros(9-len(error_per_level))))
        else:
        	scores[method] = error_per_level

print scores

x = np.arange(len(scores['counting']))
print x

plt.figure()
ax = plt.gca()
ax.bar(x-0.3, scores[methods[0]],width=0.15,color='r',align='center', label = methods[0])
ax.bar(x-0.15, scores[methods[1]],width=0.15,color='b',align='center', label = methods[1])
ax.bar(x, scores[methods[2]],width=0.15,color='g',align='center', label = methods[2])
ax.bar(x+0.15, scores[methods[3]],width=0.15,color='m',align='center', label = methods[3])
ax.bar(x+0.3, scores[methods[4]],width=0.15,color='c',align='center', label = methods[4])
ax.bar(x+0.45, scores[methods[5]],width=0.15,color='y',align='center', label = methods[5])


plt.legend(loc=2)
plt.title('Error per level')
plt.ylabel('mAE')
plt.xlabel('Levels')
plt.savefig('/var/node436/local/tstahl/vis_iep/error_per_level.pdf')

scores = {}

plt.clf()
for method in methods:
    with open('/var/scratch/spintea/Repositories/ms-caffe/output/visualization/error_per_object_%s.pickle'%(method),'rb') as handle:
    	error_per_object = pickle.load(handle)
        if len(error_per_object) < 5:
        	scores[method] = np.concatenate((error_per_object,np.zeros(9-len(error_per_object))))
        else:
        	scores[method] = error_per_object

print scores

x = np.arange(len(scores['counting']))

ax = plt.subplot(111)
ax.bar(x-0.3, scores[methods[0]],width=0.15,color='r',align='center', label = methods[0])
ax.bar(x-0.15, scores[methods[1]],width=0.15,color='b',align='center', label = methods[1])
ax.bar(x, scores[methods[2]],width=0.15,color='g',align='center', label = methods[2])
ax.bar(x+0.15, scores[methods[3]],width=0.15,color='m',align='center', label = methods[3])
ax.bar(x+0.3, scores[methods[4]],width=0.15,color='c',align='center', label = methods[4])
ax.bar(x+0.45, scores[methods[5]],width=0.15,color='y',align='center', label = methods[5])

plt.legend()
ax = plt.gca()
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[-1] = '4+'
plt.title('Error per objects')
plt.ylabel('mAE')
plt.xlabel('# of objects')
plt.savefig('/var/node436/local/tstahl/vis_iep/error_per_objects.pdf')
