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

    load_dennis = Input.Input('dennis',category,5)
    #output_pascal = Output.Output('pascal_max', category, tree_level_size, '1b')
    output_dennis = Output.Output('dennis_%s'%(pred_mode), category, 5, '1b')
    train_mat = {}
    #test_mat = {}
    for i,img_nr in enumerate(range(1,2)):
        im_dict = {}
        img_data = Data.Data(load_dennis, img_nr, 5, None)
        # we need: 
            #'image': '/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/000005.jpg'
            #'boxes' # (intersections)
            #labels
            #functions
        im_dict['image'] = '/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/%s.jpg'%(format(img_nr, "06d"))
        im_dict['boxes'] = img_data.boxes
        im_dict['labels'] = load_dennis.get_all_labels(img_nr)
        im_dict['functions'] = img_data.box_levels
        assert len(img_data.box_levels ) == len(img_data.boxes)
        #train_mat.append(im_dict)
        #train_mat[i] = im_dict
    output_dennis.save_mat(im_dict,test_mat)
    
    
if __name__ == "__main__":
    main()
