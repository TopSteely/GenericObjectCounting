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
	sgd_dummy = SGD.SGD('dennis', 'mean', 'sheep', 3, 5, 0.0001, 0.0001, 0.0001, 5)
	img_data = DummyData.DummyData()
	sgd_dummy.method(img_data)


if __name__ == "__main__":
    main()