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
import scipy.stats as stats
from scipy.stats import ttest_ind
from scipy.signal import butter, lfilter
import mne
import time as timelog
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
mne.set_log_level('WARNING')

fs=128
event_code_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6,7:7, 8:8, 9:9, 10:10, 11:11}
F_L = 4
F_H = 8

# roi_dict = {'O':['O1','Oz','O2'],
#             'T':['T7','TP7','FT7','T8','TP8','FT8'],
#             'C':['C1','C2','C3','C4','C5','C6','Cz'],
#             'CP':['CP1','CP2','CP3','CP4','CP5','CP6','CPz'],
#             'FC':['FC1','FC2','FC3','FC4','FC5','FC6','FCz'],
#             'F':['F1','F2','F3','F4','F5','F6','F7','F8','Fz'],
#             'AF':['AF3','AF4','AF7','AF8','AFz'],
#             'FP':['Fp1','Fp2']}

roi_dict = {'O':['O1','Oz','O2'],
            'T':['T7','TP7','FT7','T8','TP8','FT8'],
            'C':['C1','C2','Cz','FC1','FC2','FCz'],
            'F':['AF3','AF4','AF7','AF8','AFz','Fp1','Fp2']}

original_order = ['Fp1', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 
                  'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4',
                  'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6',
                  'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 
                  'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3',
                  'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 
                  'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']

def get_data_fc(sub_index=0,  time_range=[0,0.6]):
    data_path="./object_VMI_data/"
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
                    
    X_list = []
    y_list = []
    for fid in files:
        data = np.load(path + fid).astype(np.float32)
        event_code = int(fid.split('/')[-1].split('.')[0])
        event_class = event_code_dict[event_code]
        if  (event_code in event_code_dict.keys()):
            X = data[:,int(0.4*fs):int(2.1*fs)]
            baseline = np.mean(X[:,0:int(0.2*fs)],axis=1,keepdims=True)
            X = X - baseline
            corr_matrix = np.corrcoef(X[:,int(fs*(time_range[0]+0.2)):int(fs*(time_range[1]+0.2))].T,rowvar=False).astype(np.float32)
            X_list.append(corr_matrix)
            label = int(event_class)
            y_list.append(label)  

    X = np.array(X_list)
    y = np.array(y_list) 
    return X,y

###############compute  fc##################################
outdir = "stats/imagery_object_time-fc_sub/"
roi_list = ['O','C','T','F']
group_color = {
    'O': 'steelblue',
    'T': 'darkseagreen',
    'C': 'r',
    'F': 'darkgoldenrod'
}
os.system("mkdir -p {}".format(outdir))
for time in range(-1,6,1):
    T_L = np.round(1.0*time/10,1)
    T_H = np.round(1.0*time/10+0.1,1)
    print(f"{T_L}-{T_H}s")
    stats_list = []
    for  sub_index in range(36):
        X, y = get_data_fc(sub_index=sub_index,time_range=[T_L,T_H])
        X_mean = np.mean(X,axis=0)
        stats_list.append(X_mean)
    np.save(outdir+f"imagery_object_{F_L}-{F_H}_{T_L}_{T_H}_fc.npy",np.array(stats_list))

###############t-test fc##################################
outdir = "stats/imagery_object_time-fc_sub/"
fc_list = []
for time in range(-1,6,1):
    T_L = np.round(1.0*time/10,1)
    T_H = np.round(T_L + 0.1,1)
    data = np.load(outdir+f"imagery_object_{F_L}-{F_H}_{T_L}_{T_H}_fc.npy")
    fc_list.append(data)

roi_list = ['O','T','C','F']
ch_index_list = []
ch_list = []
for roi in roi_list:
    ch_index_list += [original_order.index(ch) for ch in  roi_dict[roi]]
    ch_list += [ch for ch in  roi_dict[roi]]

t_array = np.zeros((len(ch_index_list),len(ch_index_list)))
p_array = np.zeros((len(ch_index_list),len(ch_index_list)))
for index in range(len(ch_index_list)):
    for index2 in range(len(ch_index_list)):
        group_list = []
        for time_index in range(7):
            group_list.append(fc_list[time_index][:,index,index2])
        # f_stat, p_value = stats.f_oneway(group_list[0], group_list[1], group_list[2],group_list[3], group_list[4], group_list[5], group_list[6])
        t_stat, p_value = ttest_ind(group_list[0], group_list[2])
        t_array[index,index2] = t_stat
        p_array[index,index2] = p_value
t_array = np.nan_to_num(t_array, nan=0)
p_array = np.nan_to_num(p_array, nan=1)
np.save(outdir+f"imagery_object_{F_L}-{F_H}_fc_tvalue.npy", t_array)
np.save(outdir+f"imagery_object_{F_L}-{F_H}_fc_pvalue.npy", p_array)

t_array = np.load(outdir+f"imagery_object_{F_L}-{F_H}_fc_tvalue.npy")
p_array = np.load(outdir+f"imagery_object_{F_L}-{F_H}_fc_pvalue.npy")

fontsize = 20
sns.set(font_scale=1.2)
plt.figure(figsize=(len(ch_list), len(ch_list))) #

p_array = np.flipud(p_array)
# mask = p_array < 0.05
# p_array = -1.0 * np.log10(p_array) #* mask
ch_list2  = list(reversed(ch_list))

ax=sns.heatmap(p_array, vmin=0.01, vmax=0.1, annot=False, cmap='jet_r', square=True, xticklabels=ch_list, yticklabels=ch_list2,cbar_kws={'shrink': 0.5})
ax.tick_params(axis='x', labelsize=fontsize)
ax.tick_params(axis='y', labelsize=fontsize)
plt.yticks(rotation=0)
cbar = ax.collections[0].colorbar
cbar.set_ticks([i/100 for i in range(1,11,2)])
cbar.ax.tick_params(labelsize=fontsize)
plt.savefig(outdir+f"imagery_object_{F_L}-{F_H}_fc_-logp_value.png")
print("finished")

#######print fc p-value######
# outdir = "stats/imagery_object_time-fc_sub/"
# t_array = np.load(outdir+f"imagery_object_{F_L}-{F_H}_fc_tvalue.npy")
# p_array = np.load(outdir+f"imagery_object_{F_L}-{F_H}_fc_pvalue.npy")
#
# roi_list = ['O','T','C','F']
# ch_index_list = []
# ch_list = []
# for roi in roi_list:
#     ch_index_list += [original_order.index(ch) for ch in  roi_dict[roi]]
#     ch_list += [ch for ch in  roi_dict[roi]]
#
# print("t-value: ", t_array[ch_list.index('FC2'),ch_list.index('T8')])
# print("p-value: ", p_array[ch_list.index('FC2'),ch_list.index('T8')])