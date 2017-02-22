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

    epochs = 3

    subsamples = 500

    feature_size = 4096

    eta = math.pow(10,-5)

    for tree_level_size in range(2,4):
        #initialize
        print 'initializing', tree_level_size
        #sgd = SGD.SGD('max', category, tree_level_size, batch_size, math.pow(10,-4), 0.003, math.pow(10,-5))
        #load_pascal = Input.Input('pascal',category)
        load_dennis = Input.Input('dennis',category,tree_level_size)
        #output_pascal = Output.Output('pascal_max', category, tree_level_size, '1b')
        output_dennis_scipy = Output.Output('dennis_scipy%s'%(pred_mode), category, tree_level_size, '1b')
        output_dennis_scipy_cons = Output.Output('dennis_scipy_cons%s'%(pred_mode), category, tree_level_size, '1b')
        
        print 0,
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

        print 1,
            
        # learn SGD
        for al_i in [math.pow(10,6),math.pow(10,3),10]:#[math.pow(10,-4)]:#,math.pow(10,-2)
            for gamma_i in [math.pow(10,-5)]:#,math.pow(10,-4),math.pow(10,-3),math.pow(10,-2)
                training_loss = np.array([], dtype=np.int64).reshape(tree_level_size+1,0)
                validation_loss = np.array([], dtype=np.int64).reshape(tree_level_size+1,0)
                #sgd_pascal = SGD.SGD('pascal', 'max', category, tree_level_size, batch_size, eta_i, gamma_i, al_i)
                sgd_dennis_scipy = SGD.SGD('dennis', pred_mode, category, tree_level_size, batch_size, eta, gamma_i, al_i, feature_size)
                #sgd_dennis_scipy_cons = SGD.SGD('dennis', pred_mode, category, tree_level_size, batch_size, eta, gamma_i, al_i, feature_size)
                sgd_dennis_scipy.set_scaler(scaler_dennis)
                #sgd_dennis_scipy_cons.set_scaler(scaler_dennis)
                print 'strating ',
                start = time.time()
                sgd_dennis_scipy.learn_scipy(epochs,learn_mode,False,subsamples)
                print 'learned scipy ', time.time() - start # 10 samples = 60s, 20samples = 330s
                start = time.time()
                #sgd_dennis_scipy_cons.learn_scipy(learn_mode,True,subsamples)
                #print 'learned scipy constrained ', time.time() - start # 10 samples = 920s, 20samples = 1622s
                mse_sc,_, _ = sgd_dennis_scipy.evaluate('val_cat', subsamples)
                print 'evaluated scipy ',
                mse_tr_sc,_, _ = sgd_dennis_scipy.evaluate('train_cat', subsamples)
                #mse_sc_cons,_, _ = sgd_dennis_scipy_cons.evaluate('val_cat', subsamples)
                #print 'evaluated scipy constrained ',
                #mse_tr_sc_cons,_, _ = sgd_dennis_scipy_cons.evaluate('train_cat', subsamples)
                #preds_d_d_sc, _, level_pred_d_d_sc, max_iep_patches_d_d_sc, max_level_preds_d_d_sc = sgd_dennis_scipy.evaluate('val_cat', subsamples, True)
                #preds_d_d_sc_cons, _, level_pred_d_d_sc_cons, max_iep_patches_d_d_sc_cons, max_level_preds_d_d_sc_cons = sgd_dennis_scipy_cons.evaluate('val_cat', subsamples, True)
                

            print "Eval loss train: ", al_i, mse_tr_sc
            print "Eval loss val: ", al_i, mse_sc
    print learn_mode, pred_mode, epochs,'with scaler', debug
    
    
if __name__ == "__main__":
    main()
