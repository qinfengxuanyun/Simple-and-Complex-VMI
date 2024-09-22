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
event_code_dict = {1: 0, 2: 1}
F_L = 0.5
F_H = 4
T_L =  0
T_H = 1.5
original_order = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3',
                 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1',
                 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz',
                 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 
                 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 
                 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 
                 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 
                 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2', 'Iz']

def get_data(sub_index=0, channel='Fp1'):
    data_path ="./grating_VMI_data/"
    path = data_path + f"event_data_{F_L}-{F_H}_6s_128_bs/"
    csv_path = data_path + "data_all.csv"
    data = pd.read_csv(csv_path).values.tolist()
    subject = [i[0] for i in data]
    file_list = os.listdir(path)
    subject_now = subject[sub_index]

    files = []
    for file in file_list:
        name = int(file.split("-")[0])
        if name == subject_now:
            for file2 in os.listdir(path+file):
                files.append(file + '/' + file2)
    files = sorted(files)
    X_list = []
    y_list = []
    for fid in files:
        data = np.load(path + fid).astype(np.float32)
        events = int(fid.split("/")[-1].split("_")[0])
        label = event_code_dict[events]
        X = data[original_order.index(channel),:]
        baseline = np.mean(X[0:int(0.2*fs)],keepdims=True)
        X = X - baseline 
        X = X[int((T_L+0.2)*fs):int((T_H+0.2)*fs)]
        X_mean = np.mean(X)
        X_std = np.std(X)
        X = (X-X_mean) / (X_std+1e-12)
        X_list.append(X)
        y_list.append(label)  
        
    # X = np.array(X_list)
    # y = np.array(y_list)
    
    return X_list,y_list

#####################SVM Spatial Classifer Train and Test############################
outdir = "stats/imagery_ch_sub/"
os.system("mkdir -p {}".format(outdir))
output_file = f"{F_L}-{F_H}_imagery6s_ch_cls_{T_L}-{T_H}_sub.csv"

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
    for  sub_index in range(38):
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
            y_proba = svm.predict_proba(X_train)[:, 1]
            acc = accuracy_score(y_train, y_pred)
            auc = roc_auc_score(y_train, y_proba)

            y_pred = svm.predict(X_test)
            y_proba = svm.predict_proba(X_test)[:, 1]
            acc_test = accuracy_score(y_test, y_pred)
            auc_test = roc_auc_score(y_test, y_proba)

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

#####################Spatial Mean Decoding Acc Compute############################
for F_L,F_H in [[0.5,4],[4,8],[8,13],[13,30],[30,40]]:
    outdir = "stats/imagery_ch_sub/"
    os.system("mkdir -p {}".format(outdir))
    task = "imagery"
    file =  f"stats/imagery_ch_sub/{F_L}-{F_H}_imagery6s_ch_cls_0-1.5_sub.csv"
    file2 =  f"stats/imagery_ch_sub/{F_L}-{F_H}_imagery6s_ch_cls_0-1.5_mean_sub.csv"
    channels = pd.read_csv(file).values[:,0].tolist()
    acc_data = pd.read_csv(file).values[:,1].tolist()
    acc_data = np.array([[float(item) for item in row[1:-1].split(",")] for row in acc_data])
    acc_mean = np.mean(acc_data,axis=1).tolist()
    data = [[i,j] for i,j in zip(channels,acc_mean)]
    df = pd.DataFrame(data,columns=['ch','acc'])
    os.system("touch {}".format(file2))
    df.to_csv(file2,sep=',',index=False)
