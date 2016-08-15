#!/bin/python
import sys
import numpy as np
import time
import random
from tools import *

if len(sys.argv) < 3:
    print 'Usage: python linearSVM_L1_predict.py test_data_file_path c_model_file_path r_model_file_path c_output_file_path r_output_file_path'
    exit(1)

if __name__ == "__main__":

    test_data_file_path = sys.argv[1]
    c_model_file_path = sys.argv[2]
    r_model_file_path = sys.argv[3]
    c_output_file_path = sys.argv[4]
    r_output_file_path = sys.argv[5]

    print 'the linearSVM_L1_predict begin, the time is:'
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    print 'loading testing data.'
    test_data = readfromCSV(test_data_file_path)

    print 'preparing test_data.'
    test_feature = test_data[:,:-1] 
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

    print 'the linearSVM_L1_train end, the time is:'
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #print 'get importance ranking'
    #get_importance_ranking(clf, feature_name_file_path, output_file_path)
