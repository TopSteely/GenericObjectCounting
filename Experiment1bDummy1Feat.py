import sys
import Input
import Output
import SGD
import Data
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

    batch_size = 1

    epochs = 1

    subsamples = 1

    feature_size = 1

    eta = math.pow(10,-2)

    print 'initializing', 6
    #sgd = SGD.SGD('max', category, tree_level_size, batch_size, math.pow(10,-4), 0.003, math.pow(10,-5))
    #load_pascal = Input.Input('pascal',category)
    load_dennis = Input.Input('dennis',category,6)
    #output_pascal = Output.Output('pascal_max', category, tree_level_size, '1b')
    output_dennis = Output.Output('1feat_%s'%(pred_mode), category, 6, '1b')
    valdata = []
    trainingdata = []
    for im in load_dennis.category_val[0:subsamples]:
        valdata.append(Data.Data(load_dennis, im, 6, None, 1, True))
    for im in load_dennis.category_train[0:subsamples]:
        trainingdata.append(Data.Data(load_dennis, im, 6, None, 1, True))

    for tree_level_size in range(2,3):
        
            
        # learn SGD
        for al_i in [0.0]:#[math.pow(10,-4)]:#,math.pow(10,-2)
            for gamma_i in [math.pow(10,-3)]:#,math.pow(10,-4),math.pow(10,-3),math.pow(10,-2)
                training_loss = np.array([], dtype=np.int64).reshape(tree_level_size+1,0)
                validation_loss = np.array([], dtype=np.int64).reshape(tree_level_size+1,0)
                training_loss_old = np.array([], dtype=np.int64).reshape(tree_level_size+1,0)
                validation_loss_old = np.array([], dtype=np.int64).reshape(tree_level_size+1,0)
                #sgd_pascal = SGD.SGD('pascal', 'max', category, tree_level_size, batch_size, eta_i, gamma_i, al_i)
                sgd_dennis = SGD.SGD('dennis', pred_mode, category, tree_level_size, batch_size, eta, gamma_i, al_i,trainingdata,valdata, feature_size)
                sgd_dennis_old = SGD.SGD('dennis', 'mean', category, tree_level_size, batch_size, eta, gamma_i, al_i,trainingdata,valdata,feature_size)
                sgd_dennis_abs = SGD.SGD('dennis', 'abs', category, tree_level_size, batch_size, eta, gamma_i, al_i,trainingdata,valdata,feature_size)
                sgd_dennis_cons_pos = SGD.SGD('dennis', 'cons_pos', category, tree_level_size, batch_size, eta, gamma_i, al_i,trainingdata,valdata,feature_size)
                print al_i, gamma_i
                for epoch in range(epochs):
                    print 'epoch: ', epoch
                    #tr_l, te_l = sgd_dennis.learn('categories')
                    if debug:
                        tr_l, te_l = sgd_dennis.learn(learn_mode, subsamples, debug)
                        tr_l_old, te_l_old = sgd_dennis_old.learn(learn_mode, subsamples, debug)
                        tr_l_abs, te_l_abs = sgd_dennis_abs.learn(learn_mode, subsamples, debug)
                        tr_l_cons_pos, te_l_cons_pos = sgd_dennis_cons_pos.learn(learn_mode, subsamples, debug)

                        mse_tr,_, _ = sgd_dennis.evaluate('train_cat',subsamples)
                        mse,_, _ = sgd_dennis.evaluate('val_cat',subsamples)
                        mse_tr_old,_, _ = sgd_dennis_old.evaluate('train_cat',subsamples)
                        mse_old,_, _ = sgd_dennis_old.evaluate('val_cat',subsamples)
                        mse_tr_abs,_, _ = sgd_dennis_abs.evaluate('train_cat',subsamples)
                        mse_abs,_, _ = sgd_dennis_abs.evaluate('val_cat',subsamples)
                        mse_tr_cons_pos,_, _ = sgd_dennis_cons_pos.evaluate('train_cat',subsamples)
                        mse_cons_pos,_, _ = sgd_dennis_cons_pos.evaluate('val_cat',subsamples)
                        print 'old', tr_l_old[-1], te_l_old[-1], mse_tr, mse, sgd_dennis.w
                        print 'new', tr_l[-1], te_l[-1], mse_tr_old,mse_old, sgd_dennis_old.w
                        print 'abs', tr_l_abs[-1], te_l_abs[-1], mse_tr_abs,mse_abs, sgd_dennis_abs.w
                        print 'cons_pos', tr_l_cons_pos[-1], te_l_cons_pos[-1], mse_tr_cons_pos,mse_cons_pos, sgd_dennis_cons_pos.w
                        training_loss = np.concatenate((training_loss,tr_l), axis=1)#.reshape(-1,1)
                        validation_loss = np.concatenate((validation_loss,te_l), axis=1)#.reshape(-1,1)
                        training_loss_old = np.concatenate((training_loss_old,tr_l_old), axis=1)#.reshape(-1,1)
                        validation_loss_old = np.concatenate((validation_loss_old,te_l_old), axis=1)#.reshape(-1,1)
                    else:
                        sgd_dennis.learn(learn_mode, subsamples)
                        print 'new',sgd_dennis.w
                        sgd_dennis_old.learn(learn_mode, subsamples)
                        print 'old',sgd_dennis_old.w
                        #sgd_dennis_abs.learn(learn_mode, subsamples)
                        #print 'abs',sgd_dennis_abs.w
                        #sgd_dennis_cons_pos.learn(learn_mode, subsamples)
                        #print 'cons_pos', sgd_dennis_cons_pos.w
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
                    preds_d_d, y_d_d, level_pred_d_d, max_level_preds_d_d = sgd_dennis.evaluate('val_cat', subsamples, debug)
                    preds_d_t, y_d_t, level_pred_d_t, max_level_preds_d_t = sgd_dennis.evaluate('train_cat', subsamples, debug)
                    mse,ae, mse_non_zero = sgd_dennis.evaluate('val_cat',subsamples)
                    mse_tr,ae_tr, mse_non_zero_tr = sgd_dennis.evaluate('train_cat',subsamples)
                    print sgd_dennis.w
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
