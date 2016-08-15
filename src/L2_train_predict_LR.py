#!/bin/python
import sys
import numpy as np
import time
import random
from tools import *

if len(sys.argv) < 3:
    print 'Usage: python L2_train_predict_LR.py train_file_path test_file_path output_result_path'
    exit(1)

if __name__ == "__main__":

    train_data_file_path = sys.argv[1]
    test_data_file_path = sys.argv[2]
    output_result_path = sys.argv[3]

    print "the L2_train_predict_LR begin, the time is :"
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    print 'loading trainning data.'
    train_data = readfromCSV(train_data_file_path)

    print 'loading testing data.'
    test_data = readfromCSV(test_data_file_path)

    print 'preparing train_data.'
    train_feature = train_data[:,:-1] 
    train_label = train_data[:,-1]

    print 'preparing test_data.'
    test_feature = test_data[:,:-1] 
    test_label = test_data[:,-1]
  
    print('train: X shape: %s, Y shape: %s' % (str(train_feature.shape), str(train_label.shape)))
    print('test: X shape: %s, Y shape: %s' % (str(test_feature.shape), str(test_label.shape)))

    weight = get_weight(train_label)

    clf = fit_lr_model(train_feature, train_label, weight)

    print('predicting with lr_L2.')    
    Result = get_predict_proba(clf, test_feature)

    print ('saving the result.')
    np.savetxt(output_result_path, Result, fmt="%.5f", delimiter=',')

    print "the L2_train_predict_LR end, the time is :"
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
