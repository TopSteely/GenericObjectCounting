import sys
import Input
import Output
import Data
import SGD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import math
import numpy as np
from sklearn.svm import SVC
import time
import random

def main():
#    if len(sys.argv) != 2:
#        print 'wrong arguments - call with python Experiment1b.py <category>'
#        exit()
    category = sys.argv[1]

    learn_mode = 'all'#category_levels

    pred_mode = 'abs'

    debug = False

    batch_size = 5

    epochs = 4
    print epochs

    subsamples = 2500

    feature_size = 4096

    eta = math.pow(10,-4)

    load_dennis = Input.Input('dennis',category,20)
    #output_pascal = Output.Output('pascal_max', category, tree_level_size, '1b')
    output_dennis = Output.Output('dennis_%s'%(pred_mode), category, 20, '1b')
    train_mat = {}
    test_mat = {}
    train_mat['image'] = []
    train_mat['boxes'] = []
    train_mat['labels'] = []
    train_mat['functions'] = []
    train_mat['overlaps'] = []
    for i,img_nr in enumerate(load_dennis.training_numbers[:50]):
        print img_nr
        img_data = Data.Data(load_dennis, img_nr, 3, None)
        # we need: 
            #'image': '/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/000005.jpg'
            #'boxes' # (intersections)
            #labels
            #functions
        train_mat['image'].append('/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/%s.jpg'%(format(img_nr, "06d")))
        train_mat['boxes'].append(img_data.boxes)
        train_mat['labels'].append([load_dennis.get_all_labels(img_nr)])
        train_mat['functions'].append(img_data.box_levels)
        train_mat['overlaps'].append(img_data.gt_overlaps)
        assert len(img_data.box_levels ) == len(img_data.boxes)
        #test data:
        level_functions = np.array(img_data.box_levels)
        levels = int(np.amax(level_functions[:,1],axis=0)) + 1
        num_classes = 21
        iep = np.zeros((levels,num_classes))
        patches = img_data.gt_overlaps
        print patches.shape
        for level_index in range(levels):
            plus_boxes = np.where((level_functions[:,:]==[1,level_index]).all(axis=1))[0]
            minus_boxes = np.where((level_functions[:,:]==[-1,level_index]).all(axis=1))[0]
            level_iep = np.zeros(21)
            print plus_boxes, minus_boxes
            for c in range(num_classes):
                level_iep[c] = np.sum(patches[plus_boxes,c],axis=0)
                if c==7:
                    print level_iep[c]
                if len(minus_boxes)>0:
                    level_iep[c] += np.sum(-1 * patches[minus_boxes,c],axis=0)
                if c==7:
                    print level_iep[c]
            iep[level_index,:] = level_iep
        labels = load_dennis.get_all_labels(img_nr)
        print labels
        print iep
        test_1 = np.sum(patches[img_data.level_functions[4,:]],axis=0)
        test_2 = np.sum(patches[img_data.level_functions[5,:]],axis=0)
        print test_1
        print test_2
        print test_1-test_2
        #assert np.array_equal(iep,labels)
        raw_input()
#    for i,img_nr in enumerate(load_dennis.val_numbers):
#        print img_nr
#        img_data = Data.Data(load_dennis, img_nr, 20, None)
#        # we need: 
#            #'image': '/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/000005.jpg'
#            #'boxes' # (intersections)
#            #labels
#            #functions
#        train_mat['image'].append('/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/%s.jpg'%(format(img_nr, "06d")))
#        train_mat['boxes'].append(img_data.boxes)
#        train_mat['labels'].append([load_dennis.get_all_labels(img_nr)])
#        train_mat['functions'].append(img_data.box_levels)
#        train_mat['overlaps'].append()
#        assert len(img_data.box_levels ) == len(img_data.boxes)

    test_mat['image'] = []
    test_mat['boxes'] = []
    test_mat['labels'] = []
    test_mat['functions'] = []
    test_mat['overlaps'] = []
    for i,img_nr in enumerate(load_dennis.test_numbers[0:50]):
        print img_nr
        img_data = Data.Data(load_dennis, img_nr, 20, None)
        # we need: 
            #'image': '/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/000005.jpg'
            #'boxes' # (intersections)
            #labels
            #functions
        test_mat['image'].append('/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/%s.jpg'%(format(img_nr, "06d")))
        test_mat['boxes'].append(img_data.boxes)
        test_mat['labels'].append([load_dennis.get_all_labels(img_nr)])
        test_mat['functions'].append(img_data.box_levels)
        assert len(img_data.box_levels ) == len(img_data.boxes)
        test_mat['overlaps'].append(img_data.gt_overlaps)
    output_dennis.save_mat(train_mat,test_mat)
    
    
if __name__ == "__main__":
    main()
