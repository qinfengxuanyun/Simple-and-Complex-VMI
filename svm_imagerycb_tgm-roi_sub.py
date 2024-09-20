import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from sklearn import datasets
from sklearn.feature_selection import SelectFromModel
#from sklearn.linear_model import LassoCV, Lasso

from cuml import SVC, Lasso, LinearSVC
# from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc as skl_auc
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold,StratifiedKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import cupy as cp
import gc
import matplotlib as mpl
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.signal import butter, lfilter
import mne
import time as timelog
import random
mpl.rcParams['font.size']=10
mne.set_log_level('WARNING')

fs=128
event_code_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6,7:7, 8:8, 9:9, 10:10, 11:11}
F_L = 4
F_H = 8
roi = 'O'
# 'Visual' 'Auditory'
roi_dict = {'O':['O1','Oz','O2'],
            'T':['T7','TP7','FT7','T8','TP8','FT8'],
            'C':['C1','C2','Cz','FC1','FC2','FCz'],
            'F':['F7','F8','AF3','AF4','AF7','AF8','AFz','Fp1','Fp2']}
# roi_dict = {'O':['O1','Oz','O2'],
#             'T':['T7','TP7','FT7','T8','TP8','FT8'],
#             'C':['C1','C2','C3','C4','C5','C6','Cz'],
#             'CP':['CP1','CP2','CP3','CP4','CP5','CP6','CPz'],
#             'FC':['FC1','FC2','FC3','FC4','FC5','FC6','FCz'],
#             'F':['F1','F2','F3','F4','F5','F6','F7','F8','Fz'],
#             'AF':['AF3','AF4','AF7','AF8','AFz'],
#             'FP':['Fp1','Fp2']}
original_order = ['Fp1', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 
                  'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4',
                  'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6',
                  'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 
                  'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3',
                  'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 
                  'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']

def get_data(sub_index=0,  roi='O'):
    data_path="/home/qinfeng/imagery_cb/"
    path = data_path+f"event_data_{F_L}-{F_H}_128/" 
    subject_list = []
    for file in os.listdir(path):
        subject_list.append(file.split("_")[0])
    subject = sorted(list(set(subject_list)))
    file_list = os.listdir(path)
    subject_now = subject[sub_index]

    files = []
    for file in file_list:
        if file.split("_")[0] == subject_now:
            for file2 in os.listdir(path+file):
                event_code = int(file2.split('.')[0])
                event_class =  event_code_dict[event_code]
                if  (event_code in event_code_dict.keys()):
                    files.append(file + '/' + file2)
    files = sorted(files)

    # randnum = 8
    # random.seed(randnum)
    # random.shuffle(files)
                 
    X_list = []
    y_list = []
    for fid in files:
        data = np.load(path + fid).astype(np.float32)
        event_code = int(fid.split('/')[-1].split('.')[0])
        event_class = event_code_dict[event_code]
        if  (event_code in event_code_dict.keys()):
            ch_index = [original_order.index(channel) for channel in roi_dict[roi]]
            X = data[ch_index,int(0.4*fs):int(3.1*fs)]
            baseline = np.mean(X[:,0:int(0.2*fs)],axis=1,keepdims=True)
            X = X - baseline
            X_mean = np.mean(X,axis=1,keepdims=True)
            X_std = np.std(X,axis=1,keepdims=True)
            X = (X-X_mean) / (X_std+1e-12)
            X_list.append(X)
            label = int(event_class)
            y_list.append(label)  
        
    return np.array(X_list),np.array(y_list)                  

#################################################
outdir = "stats/imagerycb_tgm-roi_sub2/"
os.system("mkdir -p {}".format(outdir))
T_L =  0
T_H = 1.2
SUB_NUM = 36
FOLD_NUM = 10
T_RANGE = int(fs*T_H) - int(fs*T_L)

X_list = []
y_list = []
for sub_index in range(SUB_NUM):
    X, y = get_data(sub_index=sub_index,roi=roi)
    X_list.append(X)
    y_list.append(y)

test_acc_score_array = np.zeros((T_RANGE,T_RANGE))
test_auc_score_array = np.zeros((T_RANGE,T_RANGE))

for  sub_index in range(SUB_NUM):
    X, y = X_list[sub_index], y_list[sub_index]
    skf = StratifiedKFold(n_splits=FOLD_NUM) 
    test_acc_score_array_now = np.zeros((T_RANGE,T_RANGE))
    test_auc_score_array_now = np.zeros((T_RANGE,T_RANGE))
    for train_index, test_index in skf.split(X, y):
        for  time in range(T_RANGE):
            print(1.0*time/fs-T_L,'s')
            X_train, y_train= np.array([X[i][:, time +int(fs*0.2)] for i in train_index]),  np.array([y[i] for i in train_index])
            svm = LinearSVC(probability=True).fit(X_train, y_train)
            for  time2 in range(T_RANGE):
                X_test, y_test =  np.array([X[i][:, time2 +int(fs*0.2)] for i in test_index]),  np.array([y[i] for i in test_index])
                y_pred = svm.predict(X_test)
                y_proba = svm.predict_proba(X_test)
                acc_test = accuracy_score(y_test, y_pred)
                auc_test = roc_auc_score(y_test, y_proba, multi_class='ovr',average='macro')

                test_acc_score_array_now[time,time2] += acc_test
                test_auc_score_array_now[time,time2] += auc_test
                
            del svm, y_pred, y_proba
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            gc.collect()
            
    test_acc_score_array += test_acc_score_array_now / FOLD_NUM
    test_auc_score_array += test_auc_score_array_now / FOLD_NUM      
    
np.save(outdir+ f"{F_L}-{F_H}_imagerycb_time_cls_{T_L}-{T_H}_{roi}_acc.npy", test_acc_score_array/SUB_NUM)
np.save(outdir+f"{F_L}-{F_H}_imagerycb_time_cls_{T_L}-{T_H}_{roi}_auc.npy", test_auc_score_array/SUB_NUM)
        