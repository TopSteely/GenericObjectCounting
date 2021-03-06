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

    dataset = 'mscoco'

    load_dennis = Input.Input(dataset,category,20)
    sum_labels = np.zeros(81)

    if dataset == 'mscoco':
        train = load_dennis.coco_train_set.getImgIds()
        #train = train[0:5]
    else:
        train = load_dennis.training_numbers[0:1]

    print len(train)
    for i,img_nr in enumerate(train):
        #img_data = Data.Data(load_dennis, img_nr, 20, None)
        labels = load_dennis.get_all_labels(img_nr, 'train')
        sum_labels += labels
    mean_labels = sum_labels/len(train)
    mean_mean_labels = np.mean(mean_labels)


    error_0 = np.zeros(81)
    error_1 = np.zeros(81)
    error_mean = np.zeros(81)
    error_mean_mean = np.zeros(81)
    error_0_sq = np.zeros(81)
    error_1_sq = np.zeros(81)
    error_mean_sq = np.zeros(81)
    error_mean_mean_sq = np.zeros(81)
    error_0_nn = np.zeros(81)
    error_1_nn = np.zeros(81)
    error_mean_nn = np.zeros(81)
    error_mean_mean_nn = np.zeros(81)
    occurances = np.zeros(81)
    if dataset == 'mscoco':
        test = load_dennis.coco_val_set.getImgIds()
    else:
        test = load_dennis.val_numbers
    print len(test)
    for i,img_nr in enumerate(test):
        #img_data = Data.Data(load_dennis, img_nr, 20, None)
        labels = load_dennis.get_all_labels(img_nr, 'test')
        classes = np.where(labels>0)[0]
        error_0_nn[classes] += np.abs(labels[classes])
        error_1_nn[classes] += np.abs(labels[classes] - 1)
        error_mean_nn[classes] += np.abs(labels[classes] - mean_labels[classes])
        error_mean_mean_nn[classes] += np.abs(labels[classes] - mean_mean_labels)
        occurances[classes] += 1
        error_0 += np.abs(labels)
        error_0_sq += labels**2
        error_1 += np.abs(labels - 1)
        error_1_sq += (labels - 1)**2
        error_mean += np.abs(labels - mean_labels)
        error_mean_sq += (labels - mean_labels)**2
        error_mean_mean += np.abs(labels - mean_mean_labels)
        error_mean_mean_sq += (labels - mean_mean_labels)**2
    print 'error baseline 0: ', np.mean(error_0[1:]/len(test)), np.mean(error_0_nn[1:]/occurances[1:])
    print 'mRMSE 0', np.mean(np.sqrt(error_0_sq[1:]/len(test)))
    print 'mMSE 0', np.mean(error_0_sq[1:]/len(test))
    print 'error baseline 1: ', np.mean(error_1[1:]/len(test)), np.mean(error_1_nn[1:]/occurances[1:])
    print 'mRMSE 1', np.mean(np.sqrt(error_1_sq[1:]/len(test)))
    print 'mMSE 1', np.mean(error_1_sq[1:]/len(test))
    print error_1_nn
    print occurances
    print 'error baseline mean: ', np.mean(error_mean[1:]/len(test)), np.mean(error_mean_nn[1:]/occurances[1:])
    print 'mRMSE mean', np.mean(np.sqrt(error_mean_sq[1:]/len(test)))
    print 'mMSE mean', np.mean(error_mean_sq[1:]/len(test))
    print 'error baseline mean_mean: ', np.mean(error_mean_mean[1:]/len(test)), np.mean(error_mean_mean_nn[1:]/occurances[1:])
    print 'mRMSE mean mean', np.mean(np.sqrt(error_mean_mean_sq[1:]/len(test)))
    print 'mMSE mean mean', np.mean(error_mean_mean_sq[1:]/len(test))
    
    
if __name__ == "__main__":
    main()
