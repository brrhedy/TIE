#!/bin/python
import sys
import numpy as np
import time
import random
from tools import *

if len(sys.argv) < 5:
    print 'Usage: python rf_L1_train.py train_data_file_path c_model_file_path r_model_file_path feature_name_file_path output_file_path'
    exit(1)

if __name__ == "__main__":

    train_data_file_path = sys.argv[1]    
    c_model_file_path = sys.argv[2]
    r_model_file_path = sys.argv[3]
    feature_name_file_path = sys.argv[4]
    output_file_path = sys.argv[5]

    print 'the rf_L1_train begin, the time is:'
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    print 'loading trainning data.'
    train_data = readfromCSV(train_data_file_path)

    print 'preparing train_data.'
    train_feature = train_data[:,1:-1] 
    train_label = train_data[:,-1] 

    print ('X shape: %s, Y shape: %s' % (str(train_feature.shape), str(train_label.shape)))

    weight = get_weight(train_label)

    print ('classifier:')    
    clf = fit_rf_classifier_model(train_feature, train_label, weight)

    dump_model(clf, c_model_file_path)
    
    print 'get importance ranking.'
    get_importance_ranking(clf, feature_name_file_path, output_file_path)

    print ('regressor:')
    clf = fit_rf_regressor_model(train_feature, train_label, weight)

    dump_model(clf, r_model_file_path)

    print 'the rf_L1_train end, the time is:'
    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
