import sys
import Input
import Output
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

    epochs = 5

    subsamples = 40

    feature_size = 1

    eta = math.pow(10,-3)

    for tree_level_size in range(2,3):
        #initialize
        print 'initializing', tree_level_size
        #sgd = SGD.SGD('max', category, tree_level_size, batch_size, math.pow(10,-4), 0.003, math.pow(10,-5))
        #load_pascal = Input.Input('pascal',category)
        load_dennis = Input.Input('dennis',category,tree_level_size)
        #output_pascal = Output.Output('pascal_max', category, tree_level_size, '1b')
        output_dennis = Output.Output('1feat_%s'%(pred_mode), category, tree_level_size, '1b')
        
            
        # learn SGD
        for al_i in [0.0]:#[math.pow(10,-4)]:#,math.pow(10,-2)
            for gamma_i in [math.pow(10,-3)]:#,math.pow(10,-4),math.pow(10,-3),math.pow(10,-2)
                training_loss = np.array([], dtype=np.int64).reshape(tree_level_size+1,0)
                validation_loss = np.array([], dtype=np.int64).reshape(tree_level_size+1,0)
                #sgd_pascal = SGD.SGD('pascal', 'max', category, tree_level_size, batch_size, eta_i, gamma_i, al_i)
                sgd_dennis = SGD.SGD('dennis', pred_mode, category, tree_level_size, batch_size, eta, gamma_i, al_i, feature_size)
                print al_i, gamma_i
                for epoch in range(epochs):
                    #print epoch
                    #tr_l, te_l = sgd_dennis.learn('categories')
                    if debug:
                        tr_l, te_l = sgd_dennis.learn(learn_mode, subsamples, debug)
                        training_loss = np.concatenate((training_loss,tr_l), axis=1)#.reshape(-1,1)
                        validation_loss = np.concatenate((validation_loss,te_l), axis=1)#.reshape(-1,1)
                    else:
                        sgd_dennis.learn(learn_mode)
                    #print tr_l, te_l
                    
                    #training_loss.extend(tr_l)
                    #validation_loss.extend(te_l)
                    
                    #print training_loss
                    #print training_loss, validation_loss
                    #t1,_,_ = sgd_dennis.evaluate('train', 20)
                    #t2,_,_ = sgd_dennis.evaluate('val', 20)
                    #training_loss.append(t1)
                    #validation_loss.append(t2)
                    #preds_d_p, preds_skl_p, y_d_p = sgd_pascal.evaluate('train',2, True)
                    #preds_d_d, preds_skl_d, y_d_d = sgd_dennis.evaluate('train',50, True)
                    #output_pascal.plot_preds(preds_d_p, preds_skl_p, y_d_p, al_i)
                    #output_dennis.plot_preds(preds_d_d, preds_skl_d, y_d_d, al_i)
                if debug:
                    output_dennis.plot_train_val_loss(training_loss, validation_loss, eta, al_i)
            if not debug:
                if learn_mode == 'all':
                    mse,ae, mse_non_zero = sgd_dennis.evaluate('val_all')
                    mse_tr,ae_tr, mse_non_zero_tr = sgd_dennis.evaluate('train_all')
                elif learn_mode == 'category':
                    mse,ae, mse_non_zero = sgd_dennis.evaluate('val_cat')
                    mse_tr,ae_tr, mse_non_zero_tr = sgd_dennis.evaluate('train_cat')
                elif learn_mode == 'category_levels':
                    mse,ae, mse_non_zero = sgd_dennis.evaluate('val_category_levels')
                    mse_tr,ae_tr, mse_non_zero_tr = sgd_dennis.evaluate('train_category_levels')
                print "Eval loss train: ", al_i, mse_tr
                print "Eval loss val: ", al_i, mse
            else:
                if learn_mode == 'all':
                    preds_d_d, y_d_d = sgd_dennis.evaluate('val_all', subsamples, debug)
                elif learn_mode == 'category':
                    preds_d_d, y_d_d = sgd_dennis.evaluate('val_cat', subsamples, debug)
                    preds_d_t, y_d_t = sgd_dennis.evaluate('train_cat', subsamples, debug)
                    mse,ae, mse_non_zero = sgd_dennis.evaluate('val_cat',subsamples)
                    mse_tr,ae_tr, mse_non_zero_tr = sgd_dennis.evaluate('train_cat',subsamples)
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
            #output_dennis.save(mse, ae, mse_non_zero, sgd_dennis, 'ind', al_i, learn_mode)
    print learn_mode, pred_mode, epochs,'with scaler', debug
    
    
if __name__ == "__main__":
    main()
