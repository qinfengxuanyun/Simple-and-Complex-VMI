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
event_code_dict = {1: 0, 2: 1}
F_L = 0.5
F_H = 4

roi_dict = {'O':['O1','Oz','O2','Iz'],  
            'T':['T7','TP7','FT7','T8','TP8','FT8'],
            'C':['C1','C2','C3','C4','Cz','FC1','FC2','FC3','FC4']}
# roi_dict = {'O':['O1','Oz','O2','Iz'],  
#             'T':['T7','TP7','FT7','T8','TP8','FT8'],
#             'C':['C1','C2','C3','C4','C5','C6','Cz'],
#             'CP':['CP1','CP2','CP3','CP4','CP5','CP6','CPz'],
#             'FC':['FC1','FC2','FC3','FC4','FC5','FC6'],
#             'F':['F1','F2','F3','F4','F5','F6','F7','F8','Fz'],
#             'AF':['AF3','AF4','AF7','AF8','AFz'],
#             'FP':['Fp1','Fp2']}
original_order = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3',
                 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1',
                 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz',
                 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 
                 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 
                 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 
                 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 
                 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2', 'Iz']

def get_data_fc(sub_index=0, time_range=[0,0.6]):
    data_path ="/home/qinfeng/imagery_static_angle/"
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

        X = data[:, 0:int(1.7*fs)]
        baseline = np.mean(X[:,0:int(0.2*fs)],axis=1,keepdims=True)
        X = X - baseline
        corr_matrix = np.corrcoef(X[:,int(fs*(time_range[0]+0.2)):int(fs*(time_range[1]+0.2))].T,rowvar=False).astype(np.float32)
        X_list.append(corr_matrix)
        y_list.append(label)  
        
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X,y

###############compute  fc##################################
# outdir = "stats/imagery_time-fc_sub/"
# os.system("mkdir -p {}".format(outdir))

# for time in range(-1,6,1):
#     T_L = np.round(1.0*time/10,1)
#     T_H = np.round(T_L + 0.1,1)
#     stats_list = []
#     for  sub_index in range(38):
#         X, y = get_data_fc(sub_index=sub_index,time_range=[T_L,T_H])
#         X_mean = np.mean(X,axis=0)
#         stats_list.append(X_mean)
#     np.save(outdir+f"imagery_{F_L}-{F_H}_{T_L}_{T_H}_fc.npy",np.array(stats_list))

# ###############ANNOVA fc##################################
# outdir = "stats/imagery_time-fc_sub/"
# fc_list = []
# for time in range(-1,6,1):
#     T_L = np.round(1.0*time/10,1)
#     T_H = np.round(T_L + 0.1,1)
#     data = np.load(outdir+f"imagery_{F_L}-{F_H}_{T_L}_{T_H}_fc.npy")
#     fc_list.append(data)

# roi_list = ['O','T','C']
# ch_index_list = []
# ch_list = []
# for roi in roi_list:
#     ch_index_list += [original_order.index(ch) for ch in  roi_dict[roi]]
#     ch_list += [ch for ch in  roi_dict[roi]]

# t_array = np.zeros((len(ch_index_list),len(ch_index_list)))
# p_array = np.zeros((len(ch_index_list),len(ch_index_list)))
# for index in range(len(ch_index_list)):
#     for index2 in range(len(ch_index_list)):
#         group_list = []
#         for time_index in range(7):
#             group_list.append(fc_list[time_index][:,index,index2])
#         # f_stat, p_value = stats.f_oneway(group_list[0], group_list[1], group_list[2],group_list[3], group_list[4], group_list[5], group_list[6])
#         t_stat, p_value = ttest_ind(group_list[0], group_list[4])
#         t_array[index,index2] = t_stat
#         p_array[index,index2] = p_value
# t_array = np.nan_to_num(t_array, nan=0)
# p_array = np.nan_to_num(p_array, nan=1)
# np.save(outdir+f"imagery_{F_L}-{F_H}_fc_tvalue.npy",t_array)
# np.save(outdir+f"imagery_{F_L}-{F_H}_fc_pvalue.npy",p_array)

# t_array = np.load(outdir+f"imagery_{F_L}-{F_H}_fc_tvalue.npy")
# p_array = np.load(outdir+f"imagery_{F_L}-{F_H}_fc_pvalue.npy")

# # 绘制热力图
# fontsize = 20
# sns.set(font_scale=1.2)
# plt.figure(figsize=(len(ch_list), len(ch_list))) #

# p_array = np.flipud(p_array)
# # mask = p_array < 0.05
# # p_array = -1.0 * np.log10(p_array) #* mask
# ch_list2  = list(reversed(ch_list))
# p_array_max = np.max(p_array)
# ax=sns.heatmap(p_array, vmin=0.01, vmax=0.1, annot=False, cmap='jet_r', square=True, xticklabels=ch_list, yticklabels=ch_list2,cbar_kws={'shrink': 0.5})
# ax.tick_params(axis='x', labelsize=fontsize)  # 设置x轴标签的字体大小
# ax.tick_params(axis='y', labelsize=fontsize)  # 设置y轴标签的字体大小
# plt.yticks(rotation=0)
# cbar = ax.collections[0].colorbar
# cbar.set_ticks([i/100 for i in range(1,11,2)])  # 设置具体的刻度值
# # cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.3f}'))  # 自定义刻度格式为三位小数
# # 设置颜色条标签的字体大小
# cbar.ax.tick_params(labelsize=fontsize)  # 设置字体大小为12
# plt.savefig(outdir+f"imagery_{F_L}-{F_H}_fc_-logp_value.png")
# print("finished")


#######print fc p-value######
outdir = "stats/imagery_time-fc_sub/"
t_array = np.load(outdir+f"imagery_{F_L}-{F_H}_fc_tvalue.npy")
p_array = np.load(outdir+f"imagery_{F_L}-{F_H}_fc_pvalue.npy")

roi_list = ['O','T','C']
ch_index_list = []
ch_list = []
for roi in roi_list:
    ch_index_list += [original_order.index(ch) for ch in  roi_dict[roi]]
    ch_list += [ch for ch in  roi_dict[roi]]

print("t-value: ", t_array[ch_list.index('C2'),ch_list.index('Oz')])
print("p-value: ", p_array[ch_list.index('C2'),ch_list.index('Oz')])