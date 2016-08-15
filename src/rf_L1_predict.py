#!/bin/python
import sys
import numpy as np
import time
import random
from tools import *

if len(sys.argv) < 5:
    print 'Usage: python rf_L1_predict.py test_data_file_path c_model_file_path r_model_file_path c_output_file_path r_output_file_path'
    exit(1)

if __name__ == "__main__":

    test_data_file_path = sys.argv[1]
    c_model_file_path = sys.argv[2]
    r_model_file_path = sys.argv[3]
    c_output_file_path = sys.argv[4]
    r_output_file_path = sys.argv[5]
    
    print 'the rf_L1_predict begin, the time is:'
    print time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())

    print 'loading testing data.'
    test_data = readfromCSV(test_data_file_path)

    print 'preparing test_data.'
    test_feature = test_data[:,1:-1] 
    test_label = test_data[:,-1] 

    print('X shape: %s, Y shape: %s' % (str(test_feature.shape), str(test_label.shape)))

    print ('classifier:')
    clf = load_model(c_model_file_path)

    print('predicting with rf_L1.')    
    Result = get_predict(clf, test_feature)

    print ('saving the result.')
    np.savetxt(c_output_file_path, Result, fmt="%.1f", delimiter=',')

    print ('regressor:')
    clf = load_model(r_model_file_path)

    print('predicting with rf_L1.')    
    Result = get_predict(clf, test_feature)

    print ('saving the result.')
    np.savetxt(r_output_file_path, Result, fmt="%.5f", delimiter=',')

    print 'the rf_L1_predict end, the time is:'
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
