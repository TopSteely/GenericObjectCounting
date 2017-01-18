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

    batch_size = 1

    for tree_level_size in range(2,3):
        #initialize
        print 'initializing', tree_level_size
        #sgd = SGD.SGD('max', category, tree_level_size, batch_size, math.pow(10,-4), 0.003, math.pow(10,-5))
        #load_pascal = Input.Input('pascal',category)
        load_dennis = Input.Input('dennis',category)
        #output_pascal = Output.Output('pascal_max', category, tree_level_size, '1b')
        output_dennis = Output.Output('dennis_max', category, tree_level_size, '1b')
        
        print 'debugging, plot loss, compare it to scikit, !'
        
        #learn scaler
        #scaler_pascal = StandardScaler()
        training_data = load_dennis.category_train
        test_numbers_d = load_dennis.test_numbers
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
        print 'learning'
        for eta_i in [math.pow(10,-4),math.pow(10,-5)]:
	    training_loss = []
	    validation_loss = []
	    print eta_i
            for al_i in [math.pow(10,-1)]:#,math.pow(10,0),math.pow(10,-1),math.pow(10,-2)
                for gamma_i in [math.pow(10,-5)]:#,math.pow(10,-4),math.pow(10,-3),math.pow(10,-2)
                    #sgd_pascal = SGD.SGD('pascal', 'max', category, tree_level_size, batch_size, eta_i, gamma_i, al_i)
                    sgd_dennis = SGD.SGD('dennis', 'max', category, tree_level_size, batch_size, eta_i, gamma_i, al_i, 4096)
                    #sgd_pascal.set_scaler(scaler_pascal)
                    sgd_dennis.set_scaler(scaler_dennis)
                    print al_i, eta_i, gamma_i
                    for epoch in range(5):
			print epoch
                        tr_l, te_l = sgd_dennis.learn('categories', 20)
			training_loss.append(tr_l)
			validation_loss.append(te_l)
			#print training_loss, validation_loss
			#t1,_,_ = sgd_dennis.evaluate('train', 20)
			#t2,_,_ = sgd_dennis.evaluate('val', 20)
			#training_loss.append(t1)
			#validation_loss.append(t2)
            #preds_d_p, preds_skl_p, y_d_p = sgd_pascal.evaluate('train',2, True)
            #preds_d_d, preds_skl_d, y_d_d = sgd_dennis.evaluate('train',50, True)
            #output_pascal.plot_preds(preds_d_p, preds_skl_p, y_d_p, al_i)
            #output_dennis.plot_preds(preds_d_d, preds_skl_d, y_d_d, al_i)
            output_dennis.plot_train_val_loss(training_loss, validation_loss, eta_i)
            
	    # evaluate
	    print 'evaluating'
	    mse,ae, mse_non_zero = sgd_dennis.evaluate('val')
	    print mse,ae, mse_non_zero
        
            # plot/save
            print 'saving'
        
            output_dennis.save(mse, ae, mse_non_zero, sgd_dennis, eta_i)
    
    
if __name__ == "__main__":
    main()
