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
from sklearn.neural_network import MLPRegressor

def main():
#    if len(sys.argv) != 2:
#        print 'wrong arguments - call with python Experiment1b.py <category>'
#        exit()
    category = sys.argv[1]

    learn_mode = 'category'

    pred_mode = 'mean'

    batch_size = 5

    for tree_level_size in range(1,2):
        #initialize
        print 'initializing', tree_level_size
        #sgd = SGD.SGD('max', category, tree_level_size, batch_size, math.pow(10,-4), 0.003, math.pow(10,-5))
        #load_pascal = Input.Input('pascal',category)
        load_dennis = Input.Input('dennis',category)
        #output_pascal = Output.Output('pascal_max', category, tree_level_size, '1b')
        output_dennis = Output.Output('dennis_%s'%(pred_mode), category, tree_level_size, 'mlp')
        
        #learn scaler
        #scaler_pascal = StandardScaler()
        if learn_mode == 'all':
            training_data = load_dennis.training_numbers
            test_numbers_d = load_dennis.test_numbers
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
            test_numbers_d = load_dennis.category_val
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
            
        mlp_data = []
        mlp_y = []
        for img_nr in training_data:
            img_data = Data.Data(load_dennis, img_nr, 10, None)
            mlp_data.append(img_data.X[0])
            mlp_y.append(img_data.y)

        mlp_data_val = []
        mlp_y_val = []
        for img_nr in test_numbers_d:
            img_data = Data.Data(load_dennis, img_nr, 10, None)
            mlp_data_val.append(img_data.X[0])
            mlp_y_val.append(img_data.y)
        # learn SGD
        for al_i in [math.pow(10,-3),math.pow(10,-2), math.pow(10,-1)]:#[math.pow(10,-4)]:#,math.pow(10,-2)
            mlp1 = MLPRegressor(verbose=False, hidden_layer_sizes=(250,250), learning_rate='invscaling', learning_rate_init=eta_i,  alpha=al_i, activation='tanh')
            mlp2 = MLPRegressor(verbose=False, hidden_layer_sizes=(500,500), learning_rate='invscaling', learning_rate_init=eta_i,  alpha=al_i, activation='tanh')
            sgd_sklearn= SGDRegressor(eta0=math.pow(10,-4), learning_rate='invscaling', n_iter = 4)
        
        
            mlp.fit(mlp_data,mlp_y)
            sgd_sklearn.fit(scaler_dennis.transform(mlp_data),mlp_y)

        
            preds_mlp1 = mlp1.predict(mlp_data_val)
            preds_mlp1 = mlp2.predict(mlp_data_val)
            preds_sgd = sgd_sklearn.predict(scaler_dennis.transform(mlp_data_val))

            #print preds_mlp, preds_sgd, mlp_y

            print 'Mlp1: ', eta_i, al_i, act_i, np.sum(((preds_mlp1-mlp_y_val)**2)/len(mlp_y_val))
            print 'Mlp2: ', eta_i, al_i, act_i, np.sum(((preds_mlp2-mlp_y_val)**2)/len(mlp_y_val))
            print 'SKL: ', eta_i, al_i, np.sum(((preds_sgd-mlp_y_val)**2)/len(mlp_y_val))
    
    
if __name__ == "__main__":
    main()
