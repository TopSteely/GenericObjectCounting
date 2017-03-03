import numpy as np
from test_gradient_for_python_layer import test_gradient_for_python_layer

# set the inputs
input_names_and_values = [('patches_predictions', [1,0.8,0.7,0.5,0.4,0.1]), 
                          ('functions', {0:['+',0], 1:[['+',1],['+',2],['-',3]]}),
                          	('labels', [1])]

output_names = ['out1']
py_module = 'IEP_Loss_Layer'
py_layer = 'IEP_Loss_Layer'
param_str = 'some params'
propagate_down = [True, False, False]

# call the test
test_gradient_for_python_layer(input_names_and_values, output_names, 
                               py_module, py_layer, param_str, 
                               propagate_down)

# you are done!