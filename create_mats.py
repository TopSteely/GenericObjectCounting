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

    dataset = 'grid'#'dennis'

    debug = False

    batch_size = 5

    epochs = 4

    subsamples = 2500

    feature_size = 4096

    eta = math.pow(10,-4)

    load_dennis = Input.Input(dataset,category,20)
    #output_pascal = Output.Output('pascal_max', category, tree_level_size, '1b')
    output_dennis = Output.Output('%s_%s'%(dataset,pred_mode), category, 20, '1b')
    train_mat = {}
    test_mat = {}
    train_mat['image'] = []
    train_mat['boxes'] = []
    train_mat['labels'] = []
    train_mat['functions'] = []
    #train_mat['overlaps'] = []
    if dataset == 'dennis' or dataset == 'grid' or dataset == 'gt':
        train = load_dennis.training_numbers
    elif dataset == 'mscoco':
        train = load_dennis.coco_train_set.getImgIds()#catNms=load_dennis.classes
    elif dataset == 'trancos':
        train = range(404)
    print train[0:5], len(train)
    for i,img_nr in enumerate(train):
        #print i, img_nr
        img_data = Data.Data(load_dennis, img_nr, 20, None)
        # we need: 
            #'image': '/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/000005.jpg'
            #'boxes' # (intersections)
            #labels
            #functions
        if img_data.boxes != []:
            if dataset == 'dennis' or dataset == 'grid' or dataset == 'gt':
                train_mat['image'].append('/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/%s.jpg'%(format(img_nr, "06d")))
            elif dataset == 'mscoco':
                train_mat['image'].append('/var/node436/local/tstahl/mscoco/train2014/%s.jpg'%(format(img_nr, "012d")))
            elif dataset == 'trancos':
                train_mat['image'].append('/var/node436/local/tstahl/TRANCOS_v3/1-%s.jpg'%(format(img_nr, "06d")))
            train_mat['boxes'].append(img_data.boxes)
            train_mat['labels'].append([load_dennis.get_all_labels(img_nr, 'train')])
            train_mat['functions'].append(img_data.box_levels)
            #train_mat['overlaps'].append(img_data.gt_overlaps)
            assert len(img_data.box_levels ) == len(img_data.boxes)
            #test data:
            #level_functions = np.array(img_data.box_levels)
            #levels = int(np.amax(level_functions[:,1],axis=0)) + 1
        #num_classes = 21
        #iep = np.zeros((levels,num_classes))
        #print img_data.level_functions[8][0:30]
        #print img_data.level_functions[9][0:30]
        #patches = img_data.gt_overlaps
        #for level_index in range(levels):
        #    plus_boxes = np.where((level_functions[:,:]==[1,level_index]).all(axis=1))[0]
        #    minus_boxes = np.where((level_functions[:,:]==[-1,level_index]).all(axis=1))[0]
        #    level_iep = np.zeros(21)
        #    for c in range(num_classes):
        #        level_iep[c] = np.sum(patches[plus_boxes,c],axis=0)
        #        if len(minus_boxes)>0:
        #            level_iep[c] += np.sum(-1 * patches[minus_boxes,c],axis=0)
        #    iep[level_index,:] = level_iep
        #labels = load_dennis.get_all_labels(img_nr)
        #for iep_ in iep:
            #assert np.array_equal(iep_ ,labels)
    if dataset == 'trancos':
        for i,img_nr in enumerate(range(421)):
            img_data = Data.Data(load_dennis, img_nr, 20, None)
            # we need: 
                #'image': '/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/000005.jpg'
                #'boxes' # (intersections)
                #labels
                #functions
            if img_data.boxes != []:
                train_mat['image'].append('/var/node436/local/tstahl/TRANCOS_v3/2-%s.jpg'%(format(img_nr, "06d")))
                train_mat['boxes'].append(img_data.boxes)
                train_mat['labels'].append([load_dennis.get_all_labels(img_nr, 'train')])
                train_mat['functions'].append(img_data.box_levels)
                #train_mat['overlaps'].append(img_data.gt_overlaps)
                assert len(img_data.box_levels ) == len(img_data.boxes)
                #test data:
                level_functions = np.array(img_data.box_levels)
                levels = int(np.amax(level_functions[:,1],axis=0)) + 1
    if dataset == 'dennis' or dataset == 'grid' or dataset == 'gt':
        for i,img_nr in enumerate(load_dennis.val_numbers):
            print img_nr
            img_data = Data.Data(load_dennis, img_nr, 20, None)
            # we need: 
                #'image': '/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/000005.jpg'
                #'boxes' # (intersections)
                #labels
                #functions
            if dataset == 'dennis' or dataset == 'grid' or dataset == 'gt':
                train_mat['image'].append('/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/%s.jpg'%(format(img_nr, "06d")))
            elif dataset == 'mscoco':
                train_mat['image'].append('/var/node436/local/tstahl/mscoco/train2014/%s.jpg'%(format(img_nr, "012d")))
            train_mat['boxes'].append(img_data.boxes)
            train_mat['labels'].append([load_dennis.get_all_labels(img_nr,'train')])
            train_mat['functions'].append(img_data.box_levels)
            #train_mat['overlaps'].append(img_data.gt_overlaps)
            assert len(img_data.box_levels ) == len(img_data.boxes)

    test_mat['image'] = []
    test_mat['boxes'] = []
    test_mat['labels'] = []
    test_mat['functions'] = []
    #test_mat['overlaps'] = []

    if dataset == 'dennis' or dataset == 'grid' or dataset == 'gt':
        test = load_dennis.test_numbers
    elif dataset == 'mscoco':
        test = load_dennis.coco_val_set.getImgIds()
    elif dataset == 'trancos':
        test = range(424)

    for i,img_nr in enumerate(test):
        print i,img_nr
        img_data = Data.Data(load_dennis, img_nr, 20, None)
        # we need: 
            #'image': '/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/000005.jpg'
            #'boxes' # (intersections)
            #labels
            #functions
        if dataset == 'dennis' or dataset == 'grid' or dataset == 'gt':
            train_mat['image'].append('/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/%s.jpg'%(format(img_nr, "06d")))
        elif dataset == 'mscoco':
            train_mat['image'].append('/var/node436/local/tstahl/mscoco/train2014/%s.jpg'%(format(img_nr, "012d")))
        elif dataset == 'trancos':
            train_mat['image'].append('/var/node436/local/tstahl/TRANCOS_v3/3-%s.jpg'%(format(img_nr, "06d")))
        test_mat['boxes'].append(img_data.boxes)
        test_mat['labels'].append([load_dennis.get_all_labels(img_nr, 'test')])
        test_mat['functions'].append(img_data.box_levels)
        assert len(img_data.box_levels ) == len(img_data.boxes)
        #test_mat['overlaps'].append(img_data.gt_overlaps)
    output_dennis.save_mat(train_mat,test_mat, dataset)
    
    
if __name__ == "__main__":
    main()
