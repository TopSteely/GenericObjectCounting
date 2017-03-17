from scipy.io import loadmat,savemat
import numpy as np

result = loadmat('/var/scratch/spintea/Repositories/ms-caffe/data/selective_search_data/mscoco_test_toby%s_%s.mat'%(0,10000))
for from_ in [10000,20000]:#10000
    
    
    
    mat = loadmat('/var/scratch/spintea/Repositories/ms-caffe/data/selective_search_data/mscoco_test_toby%s_%s.mat'%(from_,from_+10000))
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

mat = loadmat('/var/scratch/spintea/Repositories/ms-caffe/data/selective_search_data/mscoco_test_toby%s_%s.mat'%(30000,33000))
for k in mat.keys():
        print k
        if k != '__header__' and k!= '__globals__' and k!= '__version__':
           print result[k].shape, mat[k].shape
           if k == 'image' or k == 'labels':
                result[k] = np.concatenate((result[k],(mat[k])),axis=0)
           else:
               result[k] = np.concatenate((result[k],(mat[k])),axis=1)
           print result[k].shape
mat = loadmat('/var/scratch/spintea/Repositories/ms-caffe/data/selective_search_data/mscoco_test_toby%s_%s.mat'%(33000,36000))
for k in mat.keys():
        print k
        if k != '__header__' and k!= '__globals__' and k!= '__version__':
           print result[k].shape, mat[k].shape
           if k == 'image' or k == 'labels':
                result[k] = np.concatenate((result[k],(mat[k])),axis=0)
           else:
               result[k] = np.concatenate((result[k],(mat[k])),axis=1)
           print result[k].shape
mat = loadmat('/var/scratch/spintea/Repositories/ms-caffe/data/selective_search_data/mscoco_test_toby%s_%s.mat'%(36000,39000))
for k in mat.keys():
        print k
        if k != '__header__' and k!= '__globals__' and k!= '__version__':
           print result[k].shape, mat[k].shape
           if k == 'image' or k == 'labels':
                result[k] = np.concatenate((result[k],(mat[k])),axis=0)
           else:
               result[k] = np.concatenate((result[k],(mat[k])),axis=1)
           print result[k].shape
mat = loadmat('/var/scratch/spintea/Repositories/ms-caffe/data/selective_search_data/mscoco_test_toby%s_%s.mat'%(39000,41000))
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
savemat('/var/scratch/spintea/Repositories/ms-caffe/data/selective_search_data/mscoco_test.mat',result)
