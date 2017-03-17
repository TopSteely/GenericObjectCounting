from scipy.io import loadmat,savemat
import numpy as np

result = loadmat('/var/scratch/spintea/Repositories/ms-caffe/data/selective_search_data/mscoco_train_toby%s_%s.mat'%(0,5000))
for from_ in [5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000,65000,70000,75000]:#10000
    if from_ == 75000:
        mat = loadmat('/var/scratch/spintea/Repositories/ms-caffe/data/selective_search_data/mscoco_train_toby%s_%s.mat'%(from_,from_+6000))
    else:
        mat = loadmat('/var/scratch/spintea/Repositories/ms-caffe/data/selective_search_data/mscoco_train_toby%s_%s.mat'%(from_,from_+5000))
    for k in mat.keys():
        print from_,k
        if k != '__header__' and k!= '__globals__' and k!= '__version__':
           print result[k].shape, mat[k].shape
           if k == 'image' or k == 'labels':
                result[k] = np.concatenate((result[k],(mat[k])),axis=0)
           else:
               result[k] = np.concatenate((result[k],(mat[k])),axis=1)
           print result[k].shape
    print len(result)
    print len(result['functions'])

mat = loadmat('/var/scratch/spintea/Repositories/ms-caffe/data/selective_search_data/mscoco_train_toby%s_%s.mat'%(81000,-1))
for k in mat.keys():
        print k
        if k != '__header__' and k!= '__globals__' and k!= '__version__':
           print result[k].shape, mat[k].shape
           if k == 'image' or k == 'labels':
                result[k] = np.concatenate((result[k],(mat[k])),axis=0)
           else:
               result[k] = np.concatenate((result[k],(mat[k])),axis=1)
           print result[k].shape
print len(result)
print len(result['functions'])
savemat('/var/scratch/spintea/Repositories/ms-caffe/data/selective_search_data/mscoco_train_1.mat',result)
