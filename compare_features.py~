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

    for tree_level_size in range(1,2):
        load_pascal = Input.Input('pascal',category)
        load_dennis = Input.Input('dennis',category)
        output_pascal = Output.Output('pascal_max', category, tree_level_size, '1b')
        output_dennis = Output.Output('dennis_max', category, tree_level_size, '1b')
        
        print 'plotting variance of features!'
        
        training_data = load_dennis.category_train
        test_numbers_d = load_dennis.test_numbers
	X1 = []
	X2 = []
	for img_nr in training_data[0:10]:
	     img_data = Data.Data(load_dennis, img_nr, 10, None)
	     print len(img_data.X[0])
             X1.append(img_data.X[0])
	     img_data = Data.Data(load_pascal, img_nr, 10, None)
`	     print img_data.levels[0]
	     print len(img_data.X[img_data.levels[0]])
             X2.append(img_data.X[img_data.levels[0]])

	prit len(X1), len(X2)
        print len(np.var(X1, axis=0)), len(np.var(X2, axis=0))
        print np.sum(np.var(X1, axis=0)), np.sum(np.var(X2, axis=0))
        output_dennis.plot_features_variance(np.var(X1, axis=0), np.var(X2, axis=0))
    
    
if __name__ == "__main__":
    main()
