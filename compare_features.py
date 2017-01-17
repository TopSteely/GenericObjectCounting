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
	scaler1 = StandardScaler()
	scaler2 = StandardScaler()
	for img_nr in training_data[0:10]:
	     img_data = Data.Data(load_dennis, img_nr, 10, None)
             X1.append(img_data.X[0])
	     #scaler1.partial_fit(img_data.X[0])
	     img_data = Data.Data(load_pascal, img_nr, 10, None)
             X2.append(img_data.X[img_data.levels[0][0]])
	     #scaler2.partial_fit(img_data.X[img_data.levels[0][0]])
	scaler1.fit(X1)
	scaler2.fit(X2)
        output_dennis.plot_features_variance(np.mean(scaler1.transform(X1), axis=0), np.mean(scaler2.transform(X2), axis=0))
    
    
if __name__ == "__main__":
    main()
