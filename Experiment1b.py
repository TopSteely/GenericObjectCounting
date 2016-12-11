import sys
import Input
import Output
import Data
import SGD
from sklearn.preprocessing import MinMaxScaler
import math

def main():
    if len(sys.argv) != 2:
        print 'wrong arguments - call with python Experiment1b.py <category>'
        exit()
    category = sys.argv[1]

    tree_level_size = 5

    #initialize
    sgd = SGD.SGD('max', category, 3, tree_level_size, math.pow(10,-4), 0.003, 1e-6)
    load = Input.Input('pascal',category)
    #learn scaler
    scaler = MinMaxScaler()
    training_data = load.training_numbers
    for i_img_nr, img_nr in enumerate(training_data):
        img_data = Data.Data(load, img_nr)
        scaler.partial_fit(img_data.X)
    sgd.set_scaler(scaler)
        
    # learn SGD
    for epoch in range(5):
        sgd.learn()
        
    # evaluate
    mse,ae, mse_non_zero = sgd.evaluate()
    
    # plot/save
    output = Output.Output('pascal_max', category, tree_level_size, '1b')
    output.save(mse, ae, mse_non_zero)