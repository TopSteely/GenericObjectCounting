import sys
import Input
import Output
import Data
import SGD
from sklearn.preprocessing import MinMaxScaler
import math

def main():
#    if len(sys.argv) != 2:
#        print 'wrong arguments - call with python Experiment1b.py <category>'
#        exit()
    category = sys.argv[1]

    tree_level_size = 5

    #initialize
    print 'initializing'
    sgd = SGD.SGD('max', category, 3, tree_level_size, math.pow(10,-4), 0.003, 1e-6)
    load = Input.Input('pascal',category)
    output = Output.Output('pascal_max', category, tree_level_size, '1b')
    
    #learn scaler
    scaler = MinMaxScaler()
    training_data = load.training_numbers
    for i_img_nr, img_nr in enumerate(training_data[0:50]):
        img_data = Data.Data(load, img_nr, tree_level_size)
        scaler.partial_fit(img_data.X)
    sgd.set_scaler(scaler)
        
    # learn SGD
    print 'learning'
    for epoch in range(5):
        sgd.learn()
        
    # evaluate
    print 'evaluating'
    mse,ae, mse_non_zero = sgd.evaluate()
    
    # plot/save
    print 'saving'
    
    output.save(mse, ae, mse_non_zero)
    
    
if __name__ == "__main__":
    main()