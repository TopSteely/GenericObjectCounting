from scipy.io import loadmat
import numpy as np

result = loadmat('/var/scratch/spintea/Repositories/ms-caffe/data/selective_search_data/mscoco_train_toby%s_%s.mat'%(0,5000))
for from_ in [0,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000,65000,70000,75000]:

    mat = loadmat('/var/scratch/spintea/Repositories/ms-caffe/data/selective_search_data/mscoco_train_toby%s_%s.mat'%(from_,from_+5000))
    for k in mat.keys():
    
        result[k] = np.vstack((result[k],(mat[k])))
    print len(result)
    print len(result['functions'])


