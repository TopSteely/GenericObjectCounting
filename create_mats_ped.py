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
import glob
import scipy.io

def main():
#    if len(sys.argv) != 2:
#        print 'wrong arguments - call with python Experiment1b.py <category>'
#        exit()
    category = sys.argv[1]

    learn_mode = 'all'#category_levels

    pred_mode = 'abs'

    dataset = 'CARPK'#'dennis'

    debug = False

    batch_size = 5

    epochs = 4

    subsamples = 2500

    feature_size = 4096

    eta = math.pow(10,-4)

    level_size = 20

    load_dennis = Input.Input(dataset,category,20)
    #output_pascal = Output.Output('pascal_max', category, tree_level_size, '1b')
    output_dennis = Output.Output('%s_%s'%(dataset,pred_mode), category, level_size, '1b')
    train_mat = {}
    test_mat = {}
    train_mat['image'] = []
    train_mat['boxes'] = []
    train_mat['labels'] = []
    train_mat['functions'] = []
    #train_mat['overlaps'] = []
    if dataset == 'dennis' or dataset == 'grid' or dataset == 'gt' or dataset == 'sum' or dataset == 'level':
        train = load_dennis.training_numbers
    elif dataset == 'mscoco':
        train = load_dennis.coco_train_set.getImgIds()#catNms=load_dennis.classes
        'not getting mscoc train'
    elif dataset == 'trancos':
        train = range(1,404)
    elif dataset == 'pedestrians':
        train = [7,8,9,10,11,12]
        test = [0,1,2,3,4,5,6,13,14,15,16,17,18,19]
    from_ = 0
    to_ = -1
    if dataset == 'mscoco':
        for i,img_nr in enumerate(train[from_:]):
            print i, img_nr
            if dataset != 'pedestrians':
                if dataset == 'trancos':
                    img_data = Data.Data(load_dennis, img_nr, 20, None, 1)
                else:
                    img_data = Data.Data(load_dennis, img_nr, level_size, None)
                if img_data.boxes != []:
                    if dataset == 'dennis' or dataset == 'grid' or dataset == 'gt' or dataset == 'sum' or dataset == 'level':
                        train_mat['image'].append('/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/%s.jpg'%(format(img_nr, "06d")))
                    elif dataset == 'mscoco':
                        #print img_nr, load_dennis.get_all_labels(img_nr, 'train'),[load_dennis.classes[i-1] for i in np.where(load_dennis.get_all_labels(img_nr, 'train')>0)[0]]
                        #print len(load_dennis.classes)
                        #print len(load_dennis.get_all_labels(img_nr, 'train'))
                        print load_dennis.get_all_labels(img_nr, 'train').dtype
                        train_mat['image'].append('/var/node436/local/tstahl/mscoco/train2014/%s.jpg'%(format(img_nr, "012d")))
                    elif dataset == 'trancos':
                        train_mat['image'].append('/var/node436/local/tstahl/TRANCOS_v3/1-%s.jpg'%(format(img_nr, "06d")))
                    if level_size == 1:
                        img_data.boxes = [img_data.boxes[0]]
                        img_data.box_levels = [np.array([1,0])]
                    if dataset != 'pedestrians':
                        train_mat['boxes'].append(img_data.boxes)
                        if dataset == 'trancos':
                            train_mat['labels'].append([load_dennis.get_all_labels(img_nr, 1)])
                        else:
                            train_mat['labels'].append([load_dennis.get_all_labels(img_nr, 'train')])
                        if dataset != 'sum':

                            train_mat['functions'].append(img_data.box_levels)
                            #train_mat['overlaps'].append(img_data.gt_overlaps)
                            assert len(img_data.box_levels ) == len(img_data.boxes)
            else:
                mat = scipy.io.loadmat('/var/node436/local/tstahl/ucsdpeds1/gt/vidf1_33_%s_frame_full.mat'%(format(img_nr, "03d")))
                for frame in range(1,201):
                            train_mat['image'].append('/var/node436/local/tstahl/ucsdpeds1/video/train/vidf1_33_%s.y/vidf1_33_%s_f%s.png'%(format(img_nr, "03d"),format(img_nr, "03d"),format(frame, "03d")))
                            img_data = Data.Data(load_dennis, img_nr, level_size, None, frame)
                            train_mat['boxes'].append(img_data.boxes)
                            train_mat['functions'].append(img_data.box_levels)
                            train_mat['labels'].append(len(mat['fgt']['frame'][0][0][0][frame-1][0][0][0]))
                            
                            #print len(mat['fgt']['frame'][0][0][0][frame-1][0][0][0])
                            #print img_data.box_levels
                            #raw_input()
                            
        if dataset == 'trancos':
            for i,img_nr in enumerate(range(1,421)):
                img_data = Data.Data(load_dennis, img_nr, level_size, None,2)
                # we need: 
                    #'image': '/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/000005.jpg'
                    #'boxes' # (intersections)
                    #labels
                    #functions
                if img_data.boxes != []:
                    train_mat['image'].append('/var/node436/local/tstahl/TRANCOS_v3/2-%s.jpg'%(format(img_nr, "06d")))
                    train_mat['boxes'].append(img_data.boxes)
                    train_mat['labels'].append([load_dennis.get_all_labels(img_nr, 2)])
                    train_mat['functions'].append(img_data.box_levels)
                    #train_mat['overlaps'].append(img_data.gt_overlaps)
                    assert len(img_data.box_levels ) == len(img_data.boxes)
                    #test data:
                    level_functions = np.array(img_data.box_levels)
                    levels = int(np.amax(level_functions[:,1],axis=0)) + 1
        if dataset == 'dennis' or dataset == 'grid' or dataset == 'gt' or dataset == 'sum'or dataset == 'level':
            for i,img_nr in enumerate(load_dennis.val_numbers):
                print img_nr
                img_data = Data.Data(load_dennis, img_nr, level_size, None)
                # we need: 
                    #'image': '/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/000005.jpg'
                    #'boxes' # (intersections)
                    #labels
                    #functions
                if dataset == 'dennis' or dataset == 'grid' or dataset == 'gt' or dataset == 'sum'or dataset == 'level':
                    train_mat['image'].append('/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/%s.jpg'%(format(img_nr, "06d")))
                elif dataset == 'mscoco':
                    train_mat['image'].append('/var/node436/local/tstahl/mscoco/train2014/%s.jpg'%(format(img_nr, "012d")))
                if level_size == 1:
                    img_data.boxes = [img_data.boxes[0]]
                    img_data.box_levels = [np.array([1,0])]
                #print img_data.boxes
                #print img_data.box_levels
                train_mat['boxes'].append(img_data.boxes)
                train_mat['labels'].append([load_dennis.get_all_labels(img_nr,'train')])
                
                #train_mat['overlaps'].append(img_data.gt_overlaps)
                if dataset != 'sum':
                    train_mat['functions'].append(img_data.box_levels)
                    assert len(img_data.box_levels ) == len(img_data.boxes)
        #output_dennis.save_mat(train_mat,[], dataset, from_,to_, level_size)
    else:
        with open('/var/node436/local/tstahl/datasets/%s_devkit/data/ImageSets/train.txt'%(dataset)) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        for im in content:
            print im
            if dataset == 'PUCPR+':
                train_mat['image'].append('/var/node436/local/tstahl/datasets/%s_devkit/data/Images/%s.jpg'%(dataset,im))
            elif dataset == 'CARPK':
                train_mat['image'].append('/var/node436/local/tstahl/datasets/%s_devkit/data/Images/%s.png'%(dataset,im))
            img_data = Data.Data(load_dennis, im, level_size, None, frame)
            train_mat['boxes'].append(img_data.boxes)
            train_mat['functions'].append(img_data.box_levels)
            with open('/var/node436/local/tstahl/datasets/%s_devkit/data/Annotations/%s.txt'%(dataset,im)) as f:
                annotations = f.readlines()
            train_mat['labels'].append(len(annotations))
            print 'labels', len(annotations)
        savemat('/var/scratch/spintea/Repositories/ms-caffe/data/selective_search_data/%s_train.mat'%(dataset),train_mat)

    test_mat['image'] = []
    test_mat['boxes'] = []
    test_mat['labels'] = []
    test_mat['functions'] = []
    #test_mat['overlaps'] = []

    if dataset == 'dennis' or dataset == 'grid' or dataset == 'gt' or dataset == 'sum'or dataset == 'level':
        test = load_dennis.test_numbers
    elif dataset == 'mscoco':
        test = load_dennis.coco_val_set.getImgIds()
        print len(test)
    elif dataset == 'trancos':
        test = range(1,422)

    missing = []
    error_0 = 0
    num = 0
    if dataset == 'CARPK' or dataset == 'PUCPR+':
        with open('/var/node436/local/tstahl/datasets/%s_devkit/data/ImageSets/test.txt'%(dataset)) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        for im in content:
            if dataset == 'PUCPR+':
                test_mat['image'].append('/var/node436/local/tstahl/datasets/%s_devkit/data/Images/%s.jpg'%(dataset,im))
            elif dataset == 'CARPK':
                test_mat['image'].append('/var/node436/local/tstahl/datasets/%s_devkit/data/Images/%s.png'%(dataset,im))
            img_data = Data.Data(load_dennis, im, level_size, None, frame)
            test_mat['boxes'].append(img_data.boxes)
            test_mat['functions'].append(img_data.box_levels)
            with open('/var/node436/local/tstahl/datasets/%s_devkit/data/Annotations/%s.txt'%(dataset,im)) as f:
                annotations = f.readlines()
            test_mat['labels'].append(len(annotations))
        savemat('/var/scratch/spintea/Repositories/ms-caffe/data/selective_search_data/%s_test.mat'%(dataset),test_mat)
    else:
        for i,img_nr in enumerate(test[from_:]):
            print i,img_nr
            if dataset != 'pedestrians':
                if dataset == 'trancos':
                    img_data = Data.Data(load_dennis, img_nr, 20, None, 3)
                else:
                    img_data = Data.Data(load_dennis, img_nr, level_size, None)
                if img_data.boxes == []:
                    missing.append(img_nr)
                if img_data.box_levels != []:
                    if dataset == 'dennis' or dataset == 'grid' or dataset == 'gt' or dataset == 'sum'or dataset == 'level':
                        test_mat['image'].append('/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/%s.jpg'%(format(img_nr, "06d")))
                    elif dataset == 'mscoco':
                        test_mat['image'].append('/var/node436/local/tstahl/mscoco/train2014/%s.jpg'%(format(img_nr, "012d")))
                    elif dataset == 'trancos':
                        test_mat['image'].append('/var/node436/local/tstahl/TRANCOS_v3/3-%s.jpg'%(format(img_nr, "06d")))
                    if level_size == 1:
                            img_data.boxes = [img_data.boxes[0]]
                            img_data.box_levels = [np.array([1,0])]
                    #print img_data.boxes
                    #print img_data.box_levels
                    if dataset != 'pedestrians':
                        test_mat['boxes'].append(img_data.boxes)
                        if dataset == 'trancos':
                            test_mat['labels'].append([load_dennis.get_all_labels(img_nr, 3)])
              