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

    learn_mode = 'all'

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
            test_numbers_d = load_dennis.val_numbers
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
            
        if learn_mode == 'all':
            for al_i in [math.pow(10,-4), math.pow(10,-3)]:
                mlp1 = MLPRegressor(verbose=False, hidden_layer_sizes=(2000,500), alpha=al_i, activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
                mlp2 = MLPRegressor(verbose=False, hidden_layer_sizes=(2000,1000), alpha=al_i, activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
                mlp3 = MLPRegressor(verbose=False, hidden_layer_sizes=(1000,500), alpha=al_i, activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
                mlp4 = MLPRegressor(verbose=False, hidden_layer_sizes=(2000,250), alpha=al_i, activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
                mlp5 = MLPRegressor(verbose=False, hidden_layer_sizes=(500,500), alpha=al_i, activation='tanh')#learning_rate_init=math.pow(10,-3), learning_rate='invscaling',tol=0.00001
                sgd_sklearn= SGDRegressor(eta0=math.pow(10,-4), learning_rate='invscaling', n_iter = 4)
                for bi_x in range(5):
                    mlp_data = []
                    mlp_y = []
                    for img_nr in training_data[bi_x*len(training_data)/5:(bi_x+1)*len(training_data)/5-1]:
                        img_data = Data.Data(load_dennis, img_nr, 10, None)
                        mlp_data.append(img_data.X[0])
                        mlp_y.append(img_data.y)
                    mlp1.partial_fit(scaler_dennis.transform(mlp_data),mlp_y)
                    mlp2.partial_fit(scaler_dennis.transform(mlp_data),mlp_y)
                    mlp3.partial_fit(scaler_dennis.transform(mlp_data),mlp_y)
                    mlp4.partial_fit(scaler_dennis.transform(mlp_data),mlp_y)
                    mlp5.partial_fit(scaler_dennis.transform(mlp_data),mlp_y)
                    sgd_sklearn.partial_fit(scaler_dennis.transform(mlp_data),mlp_y)

                mlp_error1 = 0.0
                mlp_error2 = 0.0
                mlp_error3 = 0.0
                mlp_error4 = 0.0
                mlp_error5 = 0.0
                sgd_error = 0.0
                for img_nr in test_numbers_d:
                    img_data = Data.Data(load_dennis, img_nr, 10, None)
                    mlp_error1 += ((mlp1.predict(np.array(img_data.X[0]).reshape(1, -1))-img_data.y)**2)
                    mlp_error2 += ((mlp2.predict(np.array(img_data.X[0]).reshape(1, -1))-img_data.y)**2)
                    mlp_error3 += ((mlp3.predict(np.array(img_data.X[0]).reshape(1, -1))-img_data.y)**2)
                    mlp_error4 += ((mlp4.predict(np.array(img_data.X[0]).reshape(1, -1))-img_data.y)**2)
                    mlp_error5 += ((mlp5.predict(np.array(img_data.X[0]).reshape(1, -1))-img_data.y)**2)
                    sgd_error += ((sgd_sklearn.predict(scaler_dennis.transform(np.array(img_data.X[0]).reshape(1, -1)))-img_data.y)**2)

                print 'Mlp1: ', al_i, mlp_error1/len(test_numbers_d)
                print 'Mlp2: ', al_i, mlp_error2/len(test_numbers_d)
                print 'Mlp3: ', al_i, mlp_error3/len(test_numbers_d)
                print 'Mlp4: ', al_i, mlp_error4/len(test_numbers_d)
                print 'Mlp5: ', al_i, mlp_error5/len(test_numbers_d)
                print 'SKL: ', al_i, sgd_error/len(test_numbers_d)

        else:
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
            for al_i in [math.pow(10,-4), math.pow(10,-3),math.pow(10,-2)]:#[math.pow(10,-4)]:#,math.pow(10,-2)
                #mlp1 = MLPRegressor(verbose=False, hidden_layer_sizes=(350,350),tol=0.00001, learning_rate='invscaling', learning_rate_init=math.pow(10,-3),  alpha=al_i, activation='tanh')
                mlp2 = MLPRegressor(verbose=False, hidden_layer_sizes=(500,500),tol=0.00001, learning_rate='invscaling', learning_rate_init=math.pow(10,-3),  alpha=al_i, activation='tanh')
                #mlp3 = MLPRegressor(verbose=False, hidden_layer_sizes=(650,650),tol=0.00001, learning_rate='invscaling', learning_rate_init=math.pow(10,-3),  alpha=al_i, activation='tanh')
                sgd_sklearn= SGDRegressor(eta0=math.pow(10,-4), learning_rate='invscaling', n_iter = 4)
            
            
                #mlp1.fit(mlp_data,mlp_y)
                mlp2.fit(mlp_data,mlp_y)
                #mlp3.fit(mlp_data,mlp_y)
                sgd_sklearn.fit(scaler_dennis.transform(mlp_data),mlp_y)

            
                #preds_mlp1 = mlp1.predict(mlp_data_val)
                preds_mlp2 = mlp2.predict(mlp_data_val)
                #preds_mlp3 = mlp3.predict(mlp_data_val)
                preds_sgd = sgd_sklearn.predict(scaler_dennis.transform(mlp_data_val))

                #print preds_mlp, preds_sgd, mlp_y

                #print 'Mlp1: ', al_i, np.sum(((preds_mlp1-mlp_y_val)**2)/len(mlp_y_val))
                print 'Mlp2: ', al_i, np.sum(((preds_mlp2-mlp_y_val)**2)/len(mlp_y_val))
                #print 'Mlp3: ', al_i, np.sum(((preds_mlp3-mlp_y_val)**2)/len(mlp_y_val))
                print 'SKL: ', al_i, np.sum(((preds_sgd-mlp_y_val)**2)/len(mlp_y_val))
    
    
if __name__ == "__main__":
    main()
