import sys
import Input
import Output
import Data
import SGD
from sklearn.preprocessing import StandardScaler
import math

def main():
#    if len(sys.argv) != 2:
#        print 'wrong arguments - call with python Experiment1b.py <category>'
#        exit()
    category = sys.argv[1]

    batch_size = 1

    for tree_level_size in range(1,2):
        #initialize
        print 'initializing'
        sgd = SGD.SGD('max', category, tree_level_size, batch_size, math.pow(10,-4), 0.003, math.pow(10,-5))
        load = Input.Input('pascal',category)
        output = Output.Output('pascal_max', category, tree_level_size, '1b')
        
        print 'debugging, plot loss, compare it to scikit, !'
        
        #learn scaler
        scaler = StandardScaler()
        training_data = load.training_numbers
        for i_img_nr, img_nr in enumerate(training_data[0:7]):
            img_data = Data.Data(load, img_nr, tree_level_size, None)
            print 'root ', img_data.levels[0][0]
            scaler.partial_fit(img_data.X[img_data.levels[0][0]])
        sgd.set_scaler(scaler)
            
        # learn SGD
        print 'learning'
        for epoch in range(55):
            sgd.learn(7)
            print sgd.evaluate('train',7)
            
        # evaluate
        print 'evaluating'
        mse,ae, mse_non_zero = sgd.evaluate('test')
        
        # plot/save
        print 'saving'
        
        output.save(mse, ae, mse_non_zero, sgd)
    
    
if __name__ == "__main__":
    main()