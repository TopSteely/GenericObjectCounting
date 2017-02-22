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

    learn_mode = 'category'

    pred_mode = 'new'

    debug = False

    batch_size = 5

    epochs = 1

    subsamples = 5

    feature_size = 4096

    eta = math.pow(10,-5)

    updates1_all = []
    updates2_all = []
    updates3_all = []

    for tree_level_size in range(1,5):
        #initialize
        print 'initializing', tree_level_size
        #sgd = SGD.SGD('max', category, tree_level_size, batch_size, math.pow(10,-4), 0.003, math.pow(10,-5))
        #load_pascal = Input.Input('pascal',category)
        load_dennis = Input.Input('dennis',category,tree_level_size)
        #output_pascal = Output.Output('pascal_max', category, tree_level_size, '1b')
        output_dennis = Output.Output('dennis_%s'%(pred_mode), category, tree_level_size, '1b')
        output_dennis_old = Output.Output('dennis_mean', category, tree_level_size, '1b')
        #learn scaler
        #scaler_pascal = StandardScaler()
        if learn_mode == 'all':
            training_data = load_dennis.training_numbers
            scaler_dennis = load_dennis.get_scaler()
            if scaler_dennis==[]:
                print "learning scaler"
                data_to_scale = []
                scaler = StandardScaler()
                print len(training_data)
                random.shuffle(training_data)
                for img_nr in training_data[0:400]:
                     img_data = Data.Data(load_dennis, img_nr, 10, None)
                     data_to_scale.extend(img_data.X)
                scaler.fit(data_to_scale)
                output_dennis.dump_scaler(scaler)
                scaler_dennis = scaler
        else:
            training_data = load_dennis.category_train
            scaler_dennis = load_dennis.get_scaler_category()
            if scaler_dennis==[]:
                print "learning scaler"
                data_to_scale = []
                scaler_category = StandardScaler()
                print len(training_data)
                random.shuffle(training_data)
                for img_nr in training_data[0:100]:
                     img_data = Data.Data(load_dennis, img_nr, 10, None)
                     data_to_scale.extend(img_data.X)
                scaler_category.fit(data_to_scale)
                output_dennis.dump_scaler_category(scaler_category)
                scaler_dennis = scaler_category
            
        # learn SGD
        for al_i in [0.01]:#[math.pow(10,-4)]:#,math.pow(10,-2)
            for gamma_i in [math.pow(10,-6)]:#,math.pow(10,-4),math.pow(10,-3),math.pow(10,-2)
                training_loss = np.array([], dtype=np.int64).reshape(tree_level_size+1,0)
                validation_loss = np.array([], dtype=np.int64).reshape(tree_level_size+1,0)
                #sgd_pascal = SGD.SGD('pascal', 'max', category, tree_level_size, batch_size, eta_i, gamma_i, al_i)
                sgd_dennis = SGD.SGD('dennis', pred_mode, category, tree_level_size, batch_size, eta, gamma_i, al_i, feature_size)
                sgd_dennis_old = SGD.SGD('dennis', 'mean', category, tree_level_size, batch_size, eta, gamma_i, al_i, feature_size)
                sgd_1_feat = SGD.SGD('dennis', pred_mode, category, tree_level_size, batch_size, eta, gamma_i, al_i, 1)
                sgd_dennis.set_scaler(scaler_dennis)
                sgd_dennis_old.set_scaler(scaler_dennis)
                for epoch in range(epochs):
                    print epoch,
                    #tr_l, te_l = sgd_dennis.learn('categories')
                    sgd_dennis_old.learn(learn_mode, subsamples)
                    sgd_dennis.learn(learn_mode, subsamples)
                    sgd_1_feat.learn(learn_mode, subsamples)

            updates1 = sgd_dennis_old.updates_all
            updates2 = sgd_dennis.updates_all
            updates3 = sgd_1_feat.updates_all

            print updates1
            print updates2
            print updates3
            updates1_all.append(updates1)
            updates2_all.append(updates2)
            updates3_all.append(updates3)
            print updates1_all
            print updates2_all
            print updates3_all

            output_dennis.plot_updates(updates1_all, updates2_all, updates3_all)
            #output_dennis.save(mse, ae, mse_non_zero, sgd_dennis, 'ind', al_i, learn_mode)
    print learn_mode, pred_mode, epochs,'with scaler', debug
    
    
if __name__ == "__main__":
    main()
