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
    for i,img_nr in enumerate(load_dennis.training_numbers):
        print img_nr
        img_data = Data.Data(load_dennis, img_nr, 20, None)
        # we need: 
            #'image': '/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/000005.jpg'
            #'boxes' # (intersections)
            #labels
            #functions
        train_mat['image'].append('/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/%s.jpg'%(format(img_nr, "06d")))
        train_mat['boxes'].append(img_data.boxes)
        train_mat['labels'].append([load_dennis.get_all_labels(img_nr)])
        train_mat['functions'].append(img_data.box_levels)
        train_mat['overlaps'].append()
        assert len(img_data.box_levels ) == len(img_data.boxes)
    for i,img_nr in enumerate(load_dennis.val_numbers):
        print img_nr
        img_data = Data.Data(load_dennis, img_nr, 20, None)
        # we need: 
            #'image': '/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/000005.jpg'
            #'boxes' # (intersections)
            #labels
            #functions
        train_mat['image'].append('/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/%s.jpg'%(format(img_nr, "06d")))
        train_mat['boxes'].append(img_data.boxes)
        train_mat['labels'].append([load_dennis.get_all_labels(img_nr)])
        train_mat['functions'].append(img_data.box_levels)
        train_mat['overlaps'].append()
        assert len(img_data.box_levels ) == len(img_data.boxes)

    test_mat['image'] = []
    test_mat['boxes'] = []
    test_mat['labels'] = []
    test_mat['functions'] = []
    test_mat['overlaps'] = []
    for i,img_nr in enumerate(load_dennis.test_numbers):
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
        test_mat['overlaps'].append()
    output_dennis.save_mat(train_mat,test_mat)
    
    
if __name__ == "__main__":
    main()
