import matplotlib
matplotlib.use('agg')
import pickle
import sys
import matplotlib.pyplot as plt

methods = ['counting','full_image','gt','grid','avg','avg_level']#trancos,mscoco,full image,'gt','grid','counting','avg','avg_level'


scores = {}
for method in methods:
    with open('/var/scratch/spintea/Repositories/ms-caffe/output/visualization/error_per_level_%s.pickle'%(method),'rb') as handle:
        error_per_level = pickle.load(handle)
        plt.plot(error_per_level,label=method)
        scores['method'] = error_per_level
y = [4, 9, 2]
z=[1,2,3]
k=[11,12,13]

ax = plt.subplot(111)
ax.bar(x-0.2, y,width=0.2,color='b',align='center')
ax.bar(x, z,width=0.2,color='g',align='center')
ax.bar(x+0.2, k,width=0.2,color='r',align='center')
ax.xaxis_date()

plt.show()

plt.legend()
plt.title('Error per level')
plt.ylabel('mAE')
plt.xlabel('Levels')
plt.savefig('/var/scratch/spintea/Repositories/ms-caffe/output/visualization/error_per_level.pdf')

plt.clf()
for method in methods:
    with open('/var/scratch/spintea/Repositories/ms-caffe/output/visualization/error_per_object_%s.pickle'%(method),'rb') as handle:
        error_per_objects = pickle.load(handle)
        plt.plot(error_per_objects,label=method)
plt.legend()
ax = plt.gca()
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[-1] = '4+'
plt.title('Error per objects')
plt.ylabel('mAE')
plt.xlabel('# of objects')
plt.savefig('/var/scratch/spintea/Repositories/ms-caffe/output/visualization/error_per_objects.pdf')
