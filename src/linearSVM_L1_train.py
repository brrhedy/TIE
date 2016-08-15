#!/bin/python
import sys
import numpy as np
import time
import random
from tools import *

if len(sys.argv) < 3:
    print 'Usage: python linearSVM_L1_train.py train_data_file_path c_model_file_path r_model_file_path'
    exit(1)

if __name__ == "__main__":

    train_data_file_path = sys.argv[1]    
    c_model_file_path = sys.argv[2]
    r_model_file_path = sys.argv[3]

    print 'the linearSVM_L1_train begin, the time is:'
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    print 'loading trainning data.'
    train_data = readfromCSV(train_data_file_path)

    print 'preparing train_data.'
    train_feature = train_data[:,:-1] 
    train_label = train_data[:,-1] 

    print ('X shape: %s, Y shape: %s' % (str(train_feature.shape), str(train_label.shape)))

    print ('classifier:')    
    clf = fit_linearSVC_model(train_feature, train_label)

    dump_model(clf, c_model_file_path)

    print ('regressor:')
    clf = fit_linearSVR_model(train_feature, train_label)

    dump_model(clf, r_model_file_path)
    
    print 'the linearSVM_L1_train end, the time is:'
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
