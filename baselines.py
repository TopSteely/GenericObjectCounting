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
    sum_labels = np.zeros(21)

    for i,img_nr in enumerate(load_dennis.training_numbers[0:10]):
        img_data = Data.Data(load_dennis, img_nr, 20, None)
        labels = load_dennis.get_all_labels(img_nr)
        sum_labels += labels
        print sum_labels
    mean_labels = sum_labels/len(load_dennis.training_numbers)
    mean_mean_labels = np.mean(mean_labels)
    print mean_labels
    print mean_mean_labels
    raw_input()


    error_0 = np.zeros(21)
    error_1 = np.zeros(21)
    error_mean = np.zeros(21)
    error_mean_mean = np.zeros(21)
    for i,img_nr in enumerate(load_dennis.val_numbers[0:10]):
        img_data = Data.Data(load_dennis, img_nr, 20, None)
        labels = load_dennis.get_all_labels(img_nr)
        error_0 += np.abs(labels)
        error_1 += np.abs(labels - 1)
        error_mean += np.abs(labels - mean_labels)
        error_mean_mean += np.abs(labels - mean_mean_labels)
    print 'error baseline 0: ', np.mean(error_0/len(load_dennis.val_numbers))
    print 'error baseline 1: ', np.mean(error_1/len(load_dennis.val_numbers))
    print 'error baseline mean: ', np.mean(error_mean/len(load_dennis.val_numbers))
    print 'error baseline mean_mean: ', np.mean(error_mean_mean/len(load_dennis.val_numbers))
    
    
if __name__ == "__main__":
    main()
