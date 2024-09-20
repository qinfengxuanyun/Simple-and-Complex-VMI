import os
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import mne
from sklearn.utils import resample
from scipy import stats
import concurrent.futures
from mne.stats import permutation_cluster_1samp_test

mpl.rcParams['font.size'] = 14
mne.set_log_level('WARNING')

#
# def permutation_test(preds, labels, num_permutations=1000):
#     observed_accuracy = accuracy_score(labels, preds)
#     permuted_accuracies = []
#
#     for _ in range(num_permutations):
#         permuted_labels = resample(labels)
#         permuted_accuracy = accuracy_score(permuted_labels, preds)
#         permuted_accuracies.append(permuted_accuracy)
#
#     p_value = np.mean(np.array(permuted_accuracies) >= observed_accuracy)
#     return p_value

def permutation_test(preds, labels, num_permutations=1000):
    observed_accuracy = accuracy_score(labels, preds)

    permuted_accuracies = []

    for _ in range(num_permutations):
        permuted_labels = resample(labels)
        permuted_accuracy = accuracy_score(permuted_labels, preds)
        permuted_accuracies.append(permuted_accuracy)

    p_value = np.mean(np.array(permuted_accuracies) >= observed_accuracy)
    return p_value

def compute_permutation_test(data):
    # Unpack the preprocessed data
    pre, label = data
    return permutation_test(pre, label)

fs = 128
# #############imagery 多ch统计(cluster permutation test）##########################
task = "imagery"
F_L = 4
F_H = 8
outdir = "stats/imagery_time-roi_sub/"
outfile = f"imagery_{F_L}-{F_H}_time_acc_cpt.png"
os.system("mkdir -p {}".format(outdir))

colors = ['steelblue', 'darkseagreen', 'r', 'darkgoldenrod']
index = 0
for roi in ['O', 'T', 'C']:  # ,[0.5,4],[4,8],[8,13]]:
    file = f"stats/imagery_time-roi_sub/{F_L}-{F_H}_imagery6s_time_cls_-0.2-1.5_{roi}_sub.csv"
    data = pd.read_csv(file).values[:, 1:]
    time_acc = data[:, 0]
    time_pre = data[:, 2]
    time_label = [int(i[0]) for i in data[0, 3][1:-1].split(", ")]
    time_s = np.array([i / fs - 0.2 for i in range(int(fs * 1.7))])

    acc = np.array([[float(item) for item in row[1:-1].split(",")] for row in time_acc])
    acc_mean = np.mean(acc, axis=1)
    # print(acc_mean)
    # acc_std = np.std(acc, axis=1)
    confidence_interval = np.array([stats.t.interval(0.95, len(acc[i])-1, loc=acc_mean[i], scale=stats.sem(acc[i])) for i in range(acc.shape[0])])
    plt.plot(time_s, acc_mean, color=colors[index], label=roi)  # navy
    plt.fill_between(time_s, confidence_interval[:,0], confidence_interval[:,1], color=colors[index], alpha=0.2)

    threshold = None  # 使用基于分布的自动阈值
    n_permutations = 10000  # 进行10000次标签打乱

    # Cluster-based permutation test
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        acc.T - 0.5,  # 数据减去 0.5
        n_permutations=n_permutations,
        threshold=threshold,
        tail=0,  # 双尾检验
        out_type='mask',
        n_jobs=32
    )
    filtered_clusters = [clusters[i] for i, p in enumerate(cluster_p_values) if p < 0.05]
    filtered_times = []
    for cluster in filtered_clusters:
        filtered_times+= time_s[cluster[0]].tolist()
    if filtered_times:
        min_time = min(filtered_times)
        max_time = max(filtered_times)
        print(f"min_time: {min_time}, max_time: {max_time}")
    else:
        print("no P value < 0.05")

    peak_value = np.max(acc_mean)
    plt.plot(filtered_times, np.array([peak_value for i in range(len(filtered_times))]), color=colors[index],
             marker='*', linestyle='none')

    index += 1

# 绘制chance level线s
channel_level = 0.5  # get_channel_level(int(file.split(".")[0].split("_")[-1]))
stim_start_time = 0
stim_end_time = 0.2
plt.axhline(y=channel_level, color='black', linestyle='--')
plt.axvline(x=stim_start_time, color='black', linestyle='--')
plt.axvline(x=stim_end_time, color='black', linestyle='--')
# plt.title('Angle Imagery')
plt.xlabel('s')
plt.ylabel('ACC')
plt.xticks(np.arange(-0.2, 1.7, 0.2))
plt.yticks(np.arange(0.48, 0.65, 0.02))
plt.legend(loc="upper right")
plt.savefig(outdir + outfile)
plt.close()

#############imagerycb 多ch统计(cluster permutation test）##########################
task = "imagerycb"
channel_level = 1/12
F_L = 4
F_H = 8
outdir = "stats/imagerycb_time-roi_sub/"
outfile = f"imagerycb_{F_L}-{F_H}_time_acc_cpt.png"
os.system("mkdir -p {}".format(outdir))
colors = ['steelblue', 'darkseagreen', 'r', 'darkgoldenrod']
index = 0
for roi in ['O','T','C','F']:  # ,[0.5,4],[4,8],[8,13]]:
    file = f"stats/imagerycb_time-roi_sub/{F_L}-{F_H}_imagerycb_time_cls_-0.2-1.5_{roi}_sub.csv"
    p_file = f"stats/imagerycb_time-roi_sub/{F_L}-{F_H}_imagerycb_time_cls_-0.2-1.5_{roi}_pvalue_sub.npy"
    data = pd.read_csv(file).values[:, 1:]
    time_acc = data[:, 0]
    time_pre = data[:, 2]
    time_label = [int(i[0]) for i in data[0, 3][1:-1].split(", ")]
    time_s = np.array([i / fs - 0.2 for i in range(int(fs * 1.7))])

    acc = np.array([[float(item) for item in row[1:-1].split(",")] for row in time_acc])
    acc_mean = np.mean(acc, axis=1)
    # print(acc_mean)
    # acc_std = np.std(acc, axis=1)
    confidence_interval = np.array([stats.t.interval(0.95, len(acc[i])-1, loc=acc_mean[i], scale=stats.sem(acc[i])) for i in range(acc.shape[0])])
    plt.plot(time_s, acc_mean, color=colors[index], label=roi)  # navy
    plt.fill_between(time_s, confidence_interval[:,0], confidence_interval[:,1], color=colors[index], alpha=0.2)

    threshold = None  # 使用基于分布的自动阈值
    n_permutations = 50000  # 进行50000次标签打乱

    # Cluster-based permutation test
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        acc.T - channel_level,  # 数据减去 1/ 12
        n_permutations=n_permutations,
        threshold=threshold,
        tail=0,  # 双尾检验
        out_type='mask',
        n_jobs=32
    )
    filtered_clusters = [clusters[i] for i, p in enumerate(cluster_p_values) if p < 0.05]
    filtered_times = []
    for cluster in filtered_clusters:
        filtered_times+= time_s[cluster[0]].tolist()
    if filtered_times:
        min_time = min(filtered_times)
        max_time = max(filtered_times)
        print(f"min_time: {min_time}, max_time: {max_time}")
    else:
        print("no P value < 0.05")

    peak_value = np.max(acc_mean)
    plt.plot(filtered_times, np.array([peak_value for i in range(len(filtered_times))]), color=colors[index],
             marker='*', linestyle='none')

    index += 1

# 绘制chance level线
stim_start_time = 0
stim_end_time = 0.55
plt.axhline(y=channel_level, color='black', linestyle='--')
plt.axvline(x=stim_start_time, color='black', linestyle='--')
plt.axvline(x=stim_end_time, color='black', linestyle='--')
# plt.title('Angle Imagery')
plt.xlabel('s')
plt.ylabel('ACC')
plt.xticks(np.arange(-0.2, 1.8, 0.2))
plt.yticks(np.arange(np.round(channel_level-0.01,2), np.round(channel_level+0.08,2), 0.01))
plt.legend(loc="upper right")
plt.savefig(outdir + outfile)
plt.close()

# #############imagery 多ch统计(permutation test）##########################
# outdir = "stats/imagery_time-roi_sub/"
# outfile = "imagery_time_acc_pt.png"
# os.system("mkdir -p {}".format(outdir))
# task = "imagery"
# F_L = 4
# F_H = 8
# colors = ['steelblue', 'darkseagreen', 'r', 'darkgoldenrod']
# index = 0
# for roi in ['O', 'T', 'C']:  # ,[0.5,4],[4,8],[8,13]]:
#     file = f"stats/imagery_time-roi_sub/{F_L}-{F_H}_imagery6s_time_cls_-0.2-1.5_{roi}_sub.csv"
#     p_file = f"stats/imagery_time-roi_sub/{F_L}-{F_H}_imagery6s_time_cls_-0.2-1.5_{roi}_pvalue_sub.npy"
#     data = pd.read_csv(file).values[:, 1:]
#     time_acc = data[:, 0]
#     time_pre = data[:, 2]
#     time_label = [int(i[0]) for i in data[0, 3][1:-1].split(", ")]
#     time_s = np.array([i / fs - 0.2 for i in range(int(fs * 1.7))])

#     acc = np.array([[float(item) for item in row[1:-1].split(",")] for row in time_acc])
#     acc_mean = np.mean(acc, axis=1)
#     # print(acc_mean)
#     # acc_std = np.std(acc, axis=1)
#     confidence_interval = np.array([stats.t.interval(0.95, len(acc[i])-1, loc=acc_mean[i], scale=stats.sem(acc[i])) for i in range(acc.shape[0])])
#     plt.plot(time_s, acc_mean, color=colors[index], label=roi)  # navy
#     plt.fill_between(time_s, confidence_interval[:,0], confidence_interval[:,1], color=colors[index], alpha=0.2)

#     # time_p = np.array([permutation_test([j[-1] for j in time_pre[i][1:-1].split(",")], time_label) for i in range(int(fs*1.7))])

#     # Preprocess the data outside of the loop
#     preprocessed_data = [([int(j[0]) for j in time_pre[i][1:-1].split(", ")], time_label) for i in range(len(time_pre))]
#     # Setup the number of workers
#     num_workers = 64  # Adjust based on your CPU
#     # Using a ThreadPoolExecutor to manage parallel processing
#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
#         # Process the permutation tests in parallel
#         results = list(executor.map(compute_permutation_test, preprocessed_data[:int(fs * 1.7)]))
#     time_p = np.array(results)
#     np.save(p_file, time_p)

#     time_p = np.load(p_file)

#     filtered_times = [i / fs - 0.2 for i, p in enumerate(time_p) if p < 0.001]
#     if filtered_times:
#         min_time = min(filtered_times)
#         max_time = max(filtered_times)
#         print(f"min_time: {min_time}, max_time: {max_time}")
#     else:
#         print("no P value < 0.001")

#     peak_value = np.max(acc_mean)
#     plt.plot(filtered_times, np.array([peak_value for i in range(len(filtered_times))]), color=colors[index],
#              marker='*', linestyle='none')

#     index += 1

# # 绘制chance level线s
# channel_level = 0.5  # get_channel_level(int(file.split(".")[0].split("_")[-1]))
# stim_time = 0
# plt.axhline(y=channel_level, color='black', linestyle='--')
# plt.axvline(x=stim_time, color='black', linestyle='--')
# # plt.title('Angle Imagery')
# plt.xlabel('s')
# plt.ylabel('ACC')
# plt.xticks(np.arange(-0.2, 1.7, 0.2))
# plt.yticks(np.arange(0.48, 0.65, 0.02))
# plt.legend(loc="upper right")
# plt.savefig(outdir + outfile)
# plt.close()

#############imagerycb 多ch统计(permutation test）##########################
# task = "imagerycb"
# F_L = 0.5
# F_H = 4
# outdir = "stats/imagerycb_time-roi_sub/"
# outfile = f"imagerycb_{F_L}-{F_H}_time_acc_pt.png"
# os.system("mkdir -p {}".format(outdir))
# colors = ['steelblue', 'darkseagreen', 'r', 'darkgoldenrod']
# index = 0
# for roi in ['O','T','C','F']:  # ,[0.5,4],[4,8],[8,13]]:
#     file = f"stats/imagerycb_time-roi_sub/{F_L}-{F_H}_imagerycb_time_cls_-0.2-1.5_{roi}_sub.csv"
#     p_file = f"stats/imagerycb_time-roi_sub/{F_L}-{F_H}_imagerycb_time_cls_-0.2-1.5_{roi}_pvalue_sub.npy"
#     data = pd.read_csv(file).values[:, 1:]
#     time_acc = data[:, 0]
#     time_pre = data[:, 2]
#     time_label = [int(i[0]) for i in data[0, 3][1:-1].split(", ")]
#     time_s = np.array([i / fs - 0.2 for i in range(int(fs * 1.7))])

#     acc = np.array([[float(item) for item in row[1:-1].split(",")] for row in time_acc])
#     acc_mean = np.mean(acc, axis=1)
#     # print(acc_mean)
#     # acc_std = np.std(acc, axis=1)
#     confidence_interval = np.array([stats.t.interval(0.95, len(acc[i])-1, loc=acc_mean[i], scale=stats.sem(acc[i])) for i in range(acc.shape[0])])
#     plt.plot(time_s, acc_mean, color=colors[index], label=roi)  # navy
#     plt.fill_between(time_s, confidence_interval[:,0], confidence_interval[:,1], color=colors[index], alpha=0.2)

#     # time_p = np.array([permutation_test([j[-1] for j in time_pre[i][1:-1].split(",")], time_label) for i in range(int(fs*1.7))])

#     # Preprocess the data outside of the loop
#     preprocessed_data = [([int(j[0]) for j in time_pre[i][1:-1].split(", ")], time_label) for i in range(len(time_pre))]
#     # Setup the number of workers
#     num_workers = 64  # Adjust based on your CPU
#     # Using a ThreadPoolExecutor to manage parallel processing
#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
#         # Process the permutation tests in parallel
#         results = list(executor.map(compute_permutation_test, preprocessed_data[:int(fs * 1.7)]))
#     time_p = np.array(results)
#     np.save(p_file, time_p)

#     # time_p = np.load(p_file)

#     filtered_times = [i / fs - 0.2 for i, p in enumerate(time_p) if p < 0.001]
#     if filtered_times:
#         min_time = min(filtered_times)
#         max_time = max(filtered_times)
#         print(f"min_time: {min_time}, max_time: {max_time}")
#     else:
#         print("no P value < 0.001")

#     peak_value = np.max(acc_mean)
#     plt.plot(filtered_times, np.array([peak_value for i in range(len(filtered_times))]), color=colors[index],
#              marker='*', linestyle='none')

#     index += 1

# # 绘制chance level线s
# channel_level = 1/12  # get_channel_level(int(file.split(".")[0].split("_")[-1]))
# stim_time = 0
# plt.axhline(y=channel_level, color='black', linestyle='--')
# plt.axvline(x=stim_time, color='black', linestyle='--')
# # plt.title('Angle Imagery')
# plt.xlabel('s')
# plt.ylabel('ACC')
# plt.xticks(np.arange(-0.2, 1.7, 0.2))
# plt.yticks(np.arange(np.round(1/12-0.01,2), np.round(1/12+0.06,2), 0.01))
# plt.legend(loc="upper right")
# plt.savefig(outdir + outfile)
# plt.close()
