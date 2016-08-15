#!/bin/python
import sys
import numpy as np
import time
import random
from tools import *

if len(sys.argv) < 4:
    print 'Usage: python lr_L1_predict.py test_data_file_path model_file_path c_output_file_path r_output_file_path' 
    exit(1)    

if __name__ == '__main__':

    test_data_file_path = sys.argv[1]
    model_file_path = sys.argv[2]
    c_output_file_path = sys.argv[3]
    r_output_file_path = sys.argv[4]

    # if (train_data_file_path == None or test_data_file_path == None):
    #     print 'file path invalid.'
    #     exit(1)

    print 'the lr_L1_predict begin, the time is:'
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    print 'loading testing data.'
    test_data = readfromCSV(test_data_file_path)

    print 'preparing test_data.'
    test_feature = test_data[:,1:-1] 
    test_label = test_data[:,-1] 

    print('X shape: %s, Y shape: %s' % (str(test_feature.shape), str(test_label.shape)))

    clf = load_model(model_file_path)
    
    print ('classifier:')
    print('predicting with lr_L1.')    
    Result = get_predict(clf, test_feature)

    print ('saving the result.')
    np.savetxt(c_output_file_path, Result, fmt="%.1f", delimiter=',')

    print ('regressor:')
    print('predicting with lr_L1.')    
    Result = get_predict_proba(clf, test_feature)

    print ('saving the result.')
    np.savetxt(r_output_file_path, Result, fmt="%.5f", delimiter=',')

    print 'the lr_L1_train end time is:'
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
