import sys
import Input
import Output
import Data
import SGD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import math
import numpy as np

def main():
#    if len(sys.argv) != 2:
#        print 'wrong arguments - call with python Experiment1b.py <category>'
#        exit()
    category = sys.argv[1]

    batch_size = 1

    for tree_level_size in range(1,2):
        #initialize
        print 'initializing'
        #sgd = SGD.SGD('max', category, tree_level_size, batch_size, math.pow(10,-4), 0.003, math.pow(10,-5))
        load = Input.Input('pascal',category)
        output = Output.Output('pascal_max', category, tree_level_size, '1b')
        
        print 'debugging, plot loss, compare it to scikit, !'
        
        #learn scaler
        scaler = StandardScaler()
        training_data = load.training_numbers
        test_numbers_d = load.test_numbers
        for i_img_nr, img_nr in enumerate(training_data[0:7]):
            img_data = Data.Data(load, img_nr, tree_level_size, None)
            print 'root ', img_data.levels[0][0]
            scaler.partial_fit(img_data.X[img_data.levels[0][0]])
        #sgd.set_scaler(scaler)
            
        # learn SGD
        print 'learning'
        for eta_i in [math.pow(10,-3),math.pow(10,-4),math.pow(10,-5),math.pow(10,-6)]:
            for al_i in [math.pow(10,2),math.pow(10,0),math.pow(10,-2),math.pow(10,-6)]:
                sgd = SGD.SGD('max', category, tree_level_size, batch_size, eta_i, 0.0003, al_i)
                sgd.set_scaler(scaler)
                print al_i, eta_i
                for epoch in range(15):
                    sgd.learn(7)
                    print sgd.evaluate('train',7)
                sgd_fut = SGDRegressor(eta0=eta_i, learning_rate='invscaling', shuffle=True, average=True, alpha=al_i, n_iter=15)
                sgd_fut_data = []
                sgd_fut_y = []
                for i_img_nr, img_nr in enumerate(training_data[0:7]):
                    img_data = Data.Data(load, img_nr, tree_level_size, scaler)
                    sgd_fut_data.append(img_data.X[img_data.levels[0][0]])
                    sgd_fut_y.append(img_data.y)
                sgd_fut.fit(sgd_fut_data, sgd_fut_y)
                sgd_fat_data = []
                sgd_fat_y = []
                for i_img_nr, img_nr in enumerate(test_numbers_d[0:7]):
                    img_data = Data.Data(load, img_nr, tree_level_size, scaler)
                    sgd_fat_data.append(img_data.X[img_data.levels[0][0]])
                    sgd_fat_y.append(img_data.y)
                sgd_a_error = ((sgd_fut.predict(sgd_fat_data) - np.array(sgd_fat_y))**2).sum()
                print 'scikit GD error: ', sgd_a_error
                
            
        # evaluate
        print 'evaluating'
        #mse,ae, mse_non_zero = sgd.evaluate('test')
        
        # plot/save
        print 'saving'
        
        #output.save(mse, ae, mse_non_zero, sgd)
    
    
if __name__ == "__main__":
    main()