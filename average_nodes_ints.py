import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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

    dataset = 'dennis'#'dennis'

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
    nodes = np.zeros(20)
    intersections = np.zeros(20)
    occurances = np.zeros(20)

    #train_mat['overlaps'] = []
    if dataset == 'dennis' or dataset == 'grid' or dataset == 'gt' or dataset == 'sum'or dataset == 'level':
        train = load_dennis.training_numbers
    from_ = 0
    to_ = 30
    if dataset != 'mscoco':
        for i,img_nr in enumerate(train[from_:to_]):
            print i, img_nr
            if dataset == 'trancos':
                img_data = Data.Data(load_dennis, img_nr, 20, None, 1)
            else:
                img_data = Data.Data(load_dennis, img_nr, level_size, None)
            for llvl in img_data.levels:
                intersections[llvl] += img_data.inters_size[llvl]
                nodes[llvl] += len(img_data.levels[llvl])
                occurances[llvl] += 1
        if dataset == 'dennis' or dataset == 'grid' or dataset == 'gt' or dataset == 'sum'or dataset == 'level':
            for i,img_nr in enumerate(load_dennis.val_numbers[from_:to_]):
                print img_nr
                img_data = Data.Data(load_dennis, img_nr, level_size, None)
                for llvl in img_data.levels:
                    intersections[llvl] += img_data.inters_size[llvl]
                    nodes[llvl] += len(img_data.levels[llvl])
                    occurances[llvl] += 1

    if dataset == 'dennis' or dataset == 'grid' or dataset == 'gt' or dataset == 'sum'or dataset == 'level':
        test = load_dennis.test_numbers

    missing = []
    for i,img_nr in enumerate(test[from_:to_]):
        print i,img_nr
        if dataset == 'trancos':
            img_data = Data.Data(load_dennis, img_nr, 20, None, 3)
        else:
            img_data = Data.Data(load_dennis, img_nr, level_size, None)

        for llvl in img_data.levels:
                intersections[llvl] += img_data.inters_size[llvl]
                nodes[llvl] += len(img_data.levels[llvl])
                occurances[llvl] += 1
    plt.figure()
    plt.plot(nodes/occurances, '-ro',label='mean Object proposals')
    plt.plot(intersections/occurances, '-bo',label='mean Intersections')
    plt.plot(occurances/occurances[0], '-go',label='Occurance')
    plt.xlabel('Image division depth')
    #plt.ylim([-1,max(max(preds,y))+1])
    #plt.xlim([-1,len(preds)+1])
    plt.legend(loc='upper right')
    plt.savefig('/var/scratch/tstahl/source/GenericObjectCounting/depth.pdf')
    
if __name__ == "__main__":
    main()
