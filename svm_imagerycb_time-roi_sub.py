import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
F_L = 4
F_H = 8
roi = 'F'
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

    # randnum = 8
    # random.seed(randnum)
    # random.shuffle(self.subject)

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
outdir = "stats/imagerycb_time-roi_sub/"
os.system("mkdir -p {}".format(outdir))
T_L = -0.2
T_H = 1.5
output_file = f"{F_L}-{F_H}_imagerycb_time_cls_{T_L}-{T_H}_{roi}_sub.csv"

feature_select = 0
tol = 0.01
alpha = 10**(-2.7)
stats_list = []

X_list = []
y_list = []
for sub_index in range(36):
    X, y = get_data(sub_index=sub_index,roi=roi)
    X_list.append(X)
    y_list.append(y)
        
for  time in range(int(fs*(T_L+0.2)), int(fs*(T_H+0.2))):
    print(1.0*time/fs-0.2,'s')
    acc_scores = []
    auc_scores = []
    test_acc_scores = []
    test_auc_scores = []
    y_pred_list = []
    y_test_list = []

    for  sub_index in range(36):
        X, y = X_list[sub_index], y_list[sub_index]
        skf = StratifiedKFold(n_splits=10)
        acc_scores_now = []
        auc_scores_now = []
        test_acc_scores_now = []
        test_auc_scores_now = []
        for train_index, test_index in skf.split(X, y):
            X_train, y_train= np.array([X[i][:, time] for i in train_index]),  np.array([y[i] for i in train_index])
            X_test, y_test =  np.array([X[i][:, time] for i in test_index]),  np.array([y[i] for i in test_index])

            svm = LinearSVC(probability=True).fit(X_train, y_train)
        
            y_pred = svm.predict(X_train)
            y_proba = svm.predict_proba(X_train)
            acc = accuracy_score(y_train, y_pred)
            auc = roc_auc_score(y_train, y_proba, multi_class='ovr',average='macro')

            y_pred = svm.predict(X_test)
            y_proba = svm.predict_proba(X_test)
            acc_test = accuracy_score(y_test, y_pred)
            auc_test = roc_auc_score(y_test, y_proba, multi_class='ovr',average='macro')
            y_pred_list += y_pred.tolist()
            y_test_list += y_test.tolist()

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
    stats_list.append([1.0*time/fs,test_acc_scores,test_auc_scores, y_pred_list,y_test_list ])

df = pd.DataFrame(stats_list,columns=['time','acc','auc','y_pred','y_test'])
os.system("touch {}".format(outdir+output_file))
df.to_csv(outdir+output_file,sep=',',index=False)

#################################################
# task = "imagerycb"
# outdir = "stats/imagerycb_time-roi/"
# os.system("mkdir -p {}".format(outdir))
# colors = ['b', 'g', 'r', 'c']
# for index, roi in enumerate(['Visual', 'Auditory']):
#     file =  f"stats/imagerycb_time-roi/{F_L}-{F_H}_imagerycb_time_cls_-0.2-1.5_{roi}.csv"
#     data = pd.read_csv(file).values[:,1].tolist()
#     data = np.array([[float(item) for item in row[1:-1].split(",")] for row in data])
#     data_mean = np.mean(data,axis=1)
#     data_std = np.std(data,axis=1)
#     time_s = np.array([i/fs-0.2 for i in range(int(fs*1.7))])
#     plt.plot(time_s,data_mean, color=colors[index],label=roi) #navy
#     plt.fill_between(time_s, data_mean - data_std, data_mean + data_std, color=colors[index], alpha=0.2)
    
# # 绘制chance level线
# channel_level = 1/12#get_channel_level(int(file.split(".")[0].split("_")[-1]))
# stim_time = 0
# plt.axhline(y=channel_level, color='black', linestyle='--')
# plt.axvline(x=stim_time, color='black', linestyle='--')
# plt.xlabel('s')
# plt.ylabel('ACC')
# plt.xticks(np.arange(-0.2, 1.7, 0.2))
# plt.yticks(np.arange(np.round(1/12,2), 0.14, 0.01))
# plt.legend(loc="upper right")
# plt.savefig(outdir+f"{task}_{F_L}-{F_H}_time_acc.png")
# plt.close()
