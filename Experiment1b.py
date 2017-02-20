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

    debug = True

    batch_size = 5

    epochs = 1

    subsamples = 2

    feature_size = 4096

    eta = math.pow(10,-5)

    for tree_level_size in range(2,3):
        #initialize
        print 'initializing', tree_level_size
        #sgd = SGD.SGD('max', category, tree_level_size, batch_size, math.pow(10,-4), 0.003, math.pow(10,-5))
        #load_pascal = Input.Input('pascal',category)
        load_dennis = Input.Input('dennis',category,tree_level_size)
        #output_pascal = Output.Output('pascal_max', category, tree_level_size, '1b')
        output_dennis = Output.Output('dennis_%s'%(pred_mode), category, tree_level_size, '1b')
        output_dennis_scipy = Output.Output('dennis_scipy%s'%(pred_mode), category, tree_level_size, '1b')
        
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
        for al_i in [0.1]:#[math.pow(10,-4)]:#,math.pow(10,-2)
            for gamma_i in [math.pow(10,-5)]:#,math.pow(10,-4),math.pow(10,-3),math.pow(10,-2)
                training_loss = np.array([], dtype=np.int64).reshape(tree_level_size+1,0)
                validation_loss = np.array([], dtype=np.int64).reshape(tree_level_size+1,0)
                training_loss_scipy = np.array([], dtype=np.int64).reshape(tree_level_size+1,0)
                validation_loss_scipy = np.array([], dtype=np.int64).reshape(tree_level_size+1,0)
                #sgd_pascal = SGD.SGD('pascal', 'max', category, tree_level_size, batch_size, eta_i, gamma_i, al_i)
                sgd_dennis = SGD.SGD('dennis', pred_mode, category, tree_level_size, batch_size, eta, gamma_i, al_i, feature_size)
                sgd_dennis_scipy = SGD.SGD('dennis', pred_mode, category, tree_level_size, batch_size, eta, gamma_i, al_i, feature_size)
                sgd_dennis.set_scaler(scaler_dennis)
                sgd_dennis_scipy.set_scaler(scaler_dennis)
                print al_i, gamma_i
                for epoch in range(epochs):
                    #print epoch
                    #tr_l, te_l = sgd_dennis.learn('categories')
                    if debug:
                        tr_l, te_l = sgd_dennis.learn(learn_mode, subsamples, debug)
                        tr_l_sc, te_l_sc = sgd_dennis_scipy.learn_scipy(learn_mode, with_constraints=False, subsamples, debug)
                        training_loss = np.concatenate((training_loss,tr_l), axis=1)#.reshape(-1,1)
                        validation_loss = np.concatenate((validation_loss,te_l), axis=1)#.reshape(-1,1)
                        training_loss_scipy = np.concatenate((training_loss_scipy,tr_l_sc), axis=1)#.reshape(-1,1)
                        validation_loss_scipy = np.concatenate((validation_loss_scipy,te_l_sc), axis=1)#.reshape(-1,1)
                    else:
                        sgd_dennis.learn(learn_mode)
                        sgd_dennis_scipy.learn_scipy(learn_mode, with_constraints=False)
                if debug:
                    output_dennis.plot_train_val_loss(training_loss, validation_loss, eta, al_i)
                    output_dennis_scipy.plot_train_val_loss(training_loss, validation_loss, eta, al_i)
            if not debug:
                if learn_mode == 'all':
                    mse,ae, mse_non_zero = sgd_dennis.evaluate('val_all')
                    mse_tr,ae_tr, mse_non_zero_tr = sgd_dennis.evaluate('train_all')
                    mse_sc,_, _ = sgd_dennis_scipy.evaluate('val_all')
                    mse_tr_sc,_, _ = sgd_dennis_scipy.evaluate('train_all')
                elif learn_mode == 'category':
                    mse,ae, mse_non_zero = sgd_dennis.evaluate('val_cat')
                    mse_tr,ae_tr, mse_non_zero_tr = sgd_dennis.evaluate('train_cat')
                    mse_sc,_, _ = sgd_dennis_scipy.evaluate('val_cat')
                    mse_tr_sc,_, _ = sgd_dennis_scipy.evaluate('train_cat')
                elif learn_mode == 'category_levels':
                    mse,ae, mse_non_zero = sgd_dennis.evaluate('val_category_levels')
                    mse_tr,ae_tr, mse_non_zero_tr = sgd_dennis.evaluate('train_category_levels')
                    mse_sc,_, _ = sgd_dennis_scipy.evaluate('val_category_levels')
                    mse_tr_sc,_, _ = sgd_dennis_scipy.evaluate('train_category_levels')
                print "Eval loss train: ", al_i, mse_tr, mse_tr_sc
                print "Eval loss val: ", al_i, mse, mse_sc
            else:
                if learn_mode == 'all':
                    preds_d_d, y_d_d = sgd_dennis.evaluate('val_all', subsamples, debug)
                elif learn_mode == 'category':
                    preds_d_d, y_d_d = sgd_dennis.evaluate('val_cat', subsamples, debug)
                    preds_d_t, y_d_t = sgd_dennis.evaluate('train_cat', subsamples, debug)
                    mse,ae, mse_non_zero = sgd_dennis.evaluate('val_cat',subsamples)
                    mse_tr,ae_tr, mse_non_zero_tr = sgd_dennis.evaluate('train_cat',subsamples)
                    preds_d_d_sc, _ = sgd_dennis_scipy.evaluate('val_cat', subsamples, debug)
                    preds_d_t_sc, _ = sgd_dennis_scipy.evaluate('train_cat', subsamples, debug)
                    mse_sc,_, _ = sgd_dennis_scipy.evaluate('val_cat',subsamples)
                    mse_tr_sc,_, _ = sgd_dennis_scipy.evaluate('train_cat',subsamples)
                    print "Eval loss train: ", al_i, mse_tr
                    print "Eval loss val: ", al_i, mse
                elif learn_mode == 'category_levels':
                    preds_d_d, y_d_d = sgd_dennis.evaluate('val_category_levels', subsamples, debug)
                    preds_d_t, y_d_t = sgd_dennis.evaluate('train_category_levels', subsamples, debug)
                    mse,ae, mse_non_zero = sgd_dennis.evaluate('val_category_levels',subsamples)
                    mse_tr,ae_tr, mse_non_zero_tr = sgd_dennis.evaluate('train_category_levels',subsamples)
                    print "Eval loss train: ", al_i, mse_tr
                    print "Eval loss val: ", al_i, mse
                output_dennis.plot_preds(preds_d_d, y_d_d, al_i, 'val')
                output_dennis.plot_preds(preds_d_t, y_d_t, al_i, 'train')
                output_dennis_scipy.plot_preds(preds_d_d_sc, y_d_d, al_i, 'val')
                output_dennis_scipy.plot_preds(preds_d_t_sc, y_d_t, al_i, 'train')
            #output_dennis.save(mse, ae, mse_non_zero, sgd_dennis, 'ind', al_i, learn_mode)
    print learn_mode, pred_mode, epochs,'with scaler', debug
    
    
if __name__ == "__main__":
    main()
