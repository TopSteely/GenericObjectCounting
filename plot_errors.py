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
        scores[method] = error_per_level

print scores

x = np.arange(len(scores['counting']))

ax = plt.subplot(111)
ax.bar(x-0.4, scores[methods[0]],width=0.2,color='r',align='center')
ax.bar(x-0.2, scores[methods[1]],width=0.2,color='b',align='center')
ax.bar(x, scores[methods[2]],width=0.2,color='g',align='center')
ax.bar(x+0.2, scores[methods[3]],width=0.2,color='m',align='center')
ax.bar(x+0.4, scores[methods[4]],width=0.2,color='c',align='center')
ax.bar(x+0.6, scores[methods[5]],width=0.2,color='y',align='center')


plt.legend()
plt.title('Error per level')
plt.ylabel('mAE')
plt.xlabel('Levels')
plt.savefig('/var/node436/local/tstahl/vis_iep/error_per_level.pdf')

plt.clf()
for method in methods:
    with open('/var/scratch/spintea/Repositories/ms-caffe/output/visualization/error_per_object_%s.pickle'%(method),'rb') as handle:
        scores['method'] = error_per_level

x = np.arange(len(scores['counting']))

ax = plt.subplot(111)
ax.bar(x-0.4, scores[methods[0]],width=0.2,color='r',align='center')
ax.bar(x-0.2, scores[methods[1]],width=0.2,color='b',align='center')
ax.bar(x, scores[methods[2]],width=0.2,color='g',align='center')
ax.bar(x+0.2, scores[methods[3]],width=0.2,color='m',align='center')
ax.bar(x+0.4, scores[methods[4]],width=0.2,color='c',align='center')
ax.bar(x+0.6, scores[methods[5]],width=0.2,color='y',align='center')

plt.legend()
ax = plt.gca()
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[-1] = '4+'
plt.title('Error per objects')
plt.ylabel('mAE')
plt.xlabel('# of objects')
plt.savefig('/var/node436/local/tstahl/vis_iep/error_per_objects.pdf')
