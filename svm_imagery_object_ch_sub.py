import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
mpl.rcParams['font.size']=10
mne.set_log_level('WARNING')

fs=128
event_code_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6,7:7, 8:8, 9:9, 10:10, 11:11}
F_L = 30
F_H = 40
T_L = 0
T_H = 1.5

original_order = ['Fp1', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 
                  'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4',
                  'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6',
                  'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 
                  'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3',
                  'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 
                  'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']

def get_data(sub_index=0,  channel='Fp1'):
    data_path="./object_VMI_data/"
    path = data_path+f"event_data_{F_L}-{F_H}_128/" 
    subject_list = []
    for file in os.listdir(path):
        subject_list.append(file.split("_")[0])
    subject = list(set(subject_list))
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

    X_list = []
    y_list = []
    for fid in files:
        data = np.load(path + fid).astype(np.float32)
        event_code = int(fid.split('/')[-1].split('.')[0])
        event_class = event_code_dict[event_code]
        if  (event_code in event_code_dict.keys()):
            X = data[original_order.index(channel), int(0.4*fs):int(3.1*fs)]
            baseline = np.mean(X[0:int(0.2*fs)],keepdims=True)
            X = X - baseline
            X = X[int((T_L+0.2)*fs):int((T_H+0.2)*fs)]
            X_mean = np.mean(X)
            X_std = np.std(X)
            X = (X-X_mean) / (X_std+1e-12)
            X_list.append(X)
            # X_mean = np.mean(X,axis=1,keepdims=True)
            # X_std = np.std(X,axis=1,keepdims=True)
            # X = (X-X_mean) / (X_std+1e-12)
            # X_list.append(X.ravel())
            label = int(event_class)
            y_list.append(label)  
        
    # return np.array(X_list),np.array(y_list)       
    return X_list,y_list
           
###################SVM Spatial Classifer Train and Test##############################
outdir = "stats/imagery_object_ch_sub/"
os.system("mkdir -p {}".format(outdir))
output_file = f"{F_L}-{F_H}_imagery_ch_cls_{T_L}-{T_H}_sub.csv"

feature_select = 0
tol = 0.01
alpha = 10**(-2.7)
stats_list = []

for  channel in original_order:
    print(channel)
    acc_scores = []
    auc_scores = []
    test_acc_scores = []
    test_auc_scores = []
    for  sub_index in range(36):
        X, y = get_data(channel=channel,sub_index=sub_index)
        skf = StratifiedKFold(n_splits=10)
        acc_scores_now = []
        auc_scores_now = []
        test_acc_scores_now = []
        test_auc_scores_now = []
        for train_index, test_index in skf.split(X, y):
            X_train, y_train= np.array([X[i] for i in train_index]),  np.array([y[i] for i in train_index])
            X_test, y_test =  np.array([X[i] for i in test_index]),  np.array([y[i] for i in test_index])
            
            svm = LinearSVC(probability=True).fit(X_train, y_train)
            y_pred = svm.predict(X_train)
            y_proba = svm.predict_proba(X_train)
            acc = accuracy_score(y_train, y_pred)
            auc = roc_auc_score(y_train, y_proba, multi_class='ovr',average='macro')

            y_pred = svm.predict(X_test)
            y_proba = svm.predict_proba(X_test)
            acc_test = accuracy_score(y_test, y_pred)
            auc_test = roc_auc_score(y_test, y_proba, multi_class='ovr',average='macro')

            acc_scores_now.append(acc)
            auc_scores_now.append(auc)
            test_acc_scores_now.append(acc_test)
            test_auc_scores_now.append(auc_test)
            
            del svm, y_pred, y_proba
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            gc.collect()

        acc_scores.append(np.mean(acc_scores_now))
        auc_scores.append(np.mean(auc_scores_now))
        test_acc_scores.append(np.mean(test_acc_scores_now))
        test_auc_scores.append(np.mean(test_auc_scores_now))
        # print(f'ACC: {np.mean(acc_scores_now)}')
        # print(f'AUC: {np.mean(auc_scores_now)}')
        # print(f'Test ACC: {np.mean(test_acc_scores_now)}')
        # print(f'Test AUC: {np.mean(test_auc_scores_now)}')


    print(f'Mean ACC: {np.mean(acc_scores)}')
    print(f'Mean AUC: {np.mean(auc_scores)}')
    print(f'Mean Test ACC: {np.mean(test_acc_scores)}')
    print(f'Mean Test AUC: {np.mean(test_auc_scores)}')
    stats_list.append([channel,test_acc_scores,test_auc_scores])

df = pd.DataFrame(stats_list,columns=['ch','acc','auc'])
os.system("touch {}".format(outdir+output_file))
df.to_csv(outdir+output_file,sep=',',index=False)

####################Spatial Mean Decoding Acc Compute#############################
for F_L,F_H in [[0.5,4],[4,8],[8,13],[13,30],[30,40]]:
    outdir = "stats/imagery_object_ch/"
    os.system("mkdir -p {}".format(outdir))
    task = "imagery"
    file =  f"stats/imagery_object_ch_sub/{F_L}-{F_H}_imagery_ch_cls_0-1.5_sub.csv"
    file2 =  f"stats/imagery_object_ch_sub/{F_L}-{F_H}_imagery_ch_cls_0-1.5_mean_sub.csv"
    channels = pd.read_csv(file).values[:,0].tolist()
    acc_data = pd.read_csv(file).values[:,1].tolist()
    acc_data = np.array([[float(item) for item in row[1:-1].split(",")] for row in acc_data])
    acc_mean = np.mean(acc_data,axis=1).tolist()
    data = [[i,j] for i,j in zip(channels,acc_mean)]
    df = pd.DataFrame(data,columns=['ch','acc'])
    os.system("touch {}".format(file2))
    df.to_csv(file2,sep=',',index=False)

