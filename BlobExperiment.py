import sys
import Input
import Output
import DummyData
import SGD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import math
import numpy as np
from sklearn.svm import SVC
import time
import random

def main():
	sgd_blob = SGD.SGD('blob', 'mean', 'sheep', 3, 5, 0.0001, 0.0001, 0.0001, 5)
	sgd_blob.learn()

if __name__ == "__main__":
    main()