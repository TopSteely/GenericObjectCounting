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

def main():
#    if len(sys.argv) != 2:
#        print 'wrong arguments - call with python Experiment1b.py <category>'
#        exit()
    category = sys.argv[1]

    batch_size = 1

    for tree_level_size in range(2,3):
        #initialize
        print 'initializing'
	#todo: visualize change of features per windows/full image
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
		scaler_category = StandardScaler()
		print len(training_data)
		for i_img_nr, img_nr in enumerate(training_data):
		     img_data = Data.Data(load_dennis, img_nr, tree_level_size, None)
		     scaler_category.partial_fit(img_data.X)
		output_dennis.dump_scaler_category(scaler_category)
            
        # learn SGD
        print 'learning'
        for eta_i in [math.pow(10,-4)]:#,math.pow(10,-5)
            for al_i in [math.pow(10,-1)]:#,math.pow(10,0),math.pow(10,-1),math.pow(10,-2)
                for gamma_i in [math.pow(10,-5)]:#,math.pow(10,-4),math.pow(10,-3),math.pow(10,-2)
                    #sgd_pascal = SGD.SGD('pascal', 'max', category, tree_level_size, batch_size, eta_i, gamma_i, al_i)
                    sgd_dennis = SGD.SGD('dennis', 'max', category, tree_level_size, batch_size, eta_i, gamma_i, al_i, 4096)
                    #sgd_pascal.set_scaler(scaler_pascal)
                    sgd_dennis.set_scaler(scaler_dennis)
                    print al_i, eta_i, gamma_i
                    for epoch in range(1):
                        #sgd_pascal.learn(1)
                        sgd_dennis.learn('categories')
                    #preds_d_p, preds_skl_p, y_d_p = sgd_pascal.evaluate('train',2, True)
                    preds_d_d, preds_skl_d, y_d_d = sgd_dennis.evaluate('train',88, True)
                    #output_pascal.plot_preds(preds_d_p, preds_skl_p, y_d_p, al_i)
                    #output_dennis.plot_preds(preds_d_d, preds_skl_d, y_d_d, al_i)
#                    sgd_fut = SGDRegressor(eta0=eta_i, learning_rate='invscaling', shuffle=True, average=True, alpha=al_i, n_iter=15)
#                    sgd_fut_data = []
#                    sgd_fut_y = []
#                    for i_img_nr, img_nr in enumerate(training_data[0:7]):
#                        img_data = Data.Data(load_pascal, img_nr, tree_level_size, scaler_pascal)
#                        sgd_fut_data.append(img_data.X[img_data.levels[0][0]])
#                        sgd_fut_y.append(img_data.y)
#                    sgd_fut.fit(sgd_fut_data, sgd_fut_y)
#                    sgd_fat_data = []
#                    sgd_fat_y = []
#                    for i_img_nr, img_nr in enumerate(test_numbers_d[0:7]):
#                        img_data = Data.Data(load_pascal, img_nr, tree_level_size, scaler_pascal)
#                        sgd_fat_data.append(img_data.X[img_data.levels[0][0]])
#                        sgd_fat_y.append(img_data.y)
#                    sgd_a_error = ((sgd_fut.predict(sgd_fat_data) - np.array(sgd_fat_y))**2)#.sum()
#                    print 'scikit GD error: ', sgd_fut.predict(sgd_fat_data), sgd_a_error, np.array(sgd_fat_y)
                    
                
            
        # evaluate
        print 'evaluating'
        mse,ae, mse_non_zero = sgd_dennis.evaluate('test')
        
        # plot/save
        print 'saving'
        
        output_dennis.save(mse, ae, mse_non_zero, sgd)
    
    
if __name__ == "__main__":
    main()
