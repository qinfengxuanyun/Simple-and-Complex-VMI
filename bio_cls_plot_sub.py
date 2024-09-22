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
# #############imagery multi channel(cluster permutation test）##########################
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
    confidence_interval = np.array([stats.t.interval(0.95, len(acc[i])-1, loc=acc_mean[i], scale=stats.sem(acc[i])) for i in range(acc.shape[0])])
    plt.plot(time_s, acc_mean, color=colors[index], label=roi)  # navy
    plt.fill_between(time_s, confidence_interval[:,0], confidence_interval[:,1], color=colors[index], alpha=0.2)

    threshold = None
    n_permutations = 10000

    # Cluster-based permutation test
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        acc.T - 0.5,
        n_permutations=n_permutations,
        threshold=threshold,
        tail=0,
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

# plot chance level line
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

#############imagery_object multi channel(cluster permutation test）##########################
task = "imagery_object"
channel_level = 1/12
F_L = 4
F_H = 8
outdir = "stats/imagery_object_time-roi_sub/"
outfile = f"imagery_object_{F_L}-{F_H}_time_acc_cpt.png"
os.system("mkdir -p {}".format(outdir))
colors = ['steelblue', 'darkseagreen', 'r', 'darkgoldenrod']
index = 0
for roi in ['O','T','C','F']:  # ,[0.5,4],[4,8],[8,13]]:
    file = f"stats/imagery_object_time-roi_sub/{F_L}-{F_H}_imagery_object_time_cls_-0.2-1.5_{roi}_sub.csv"
    p_file = f"stats/imagery_object_time-roi_sub/{F_L}-{F_H}_imagery_object_time_cls_-0.2-1.5_{roi}_pvalue_sub.npy"
    data = pd.read_csv(file).values[:, 1:]
    time_acc = data[:, 0]
    time_pre = data[:, 2]
    time_label = [int(i[0]) for i in data[0, 3][1:-1].split(", ")]
    time_s = np.array([i / fs - 0.2 for i in range(int(fs * 1.7))])

    acc = np.array([[float(item) for item in row[1:-1].split(",")] for row in time_acc])
    acc_mean = np.mean(acc, axis=1)
    confidence_interval = np.array([stats.t.interval(0.95, len(acc[i])-1, loc=acc_mean[i], scale=stats.sem(acc[i])) for i in range(acc.shape[0])])
    plt.plot(time_s, acc_mean, color=colors[index], label=roi)  # navy
    plt.fill_between(time_s, confidence_interval[:,0], confidence_interval[:,1], color=colors[index], alpha=0.2)

    threshold = None
    n_permutations = 50000

    # Cluster-based permutation test
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        acc.T - channel_level,
        n_permutations=n_permutations,
        threshold=threshold,
        tail=0,
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

# plot chance level line
stim_start_time = 0
stim_end_time = 0.55
plt.axhline(y=channel_level, color='black', linestyle='--')
plt.axvline(x=stim_start_time, color='black', linestyle='--')
plt.axvline(x=stim_end_time, color='black', linestyle='--')
plt.xlabel('s')
plt.ylabel('ACC')
plt.xticks(np.arange(-0.2, 1.8, 0.2))
plt.yticks(np.arange(np.round(channel_level-0.01,2), np.round(channel_level+0.08,2), 0.01))
plt.legend(loc="upper right")
plt.savefig(outdir + outfile)
plt.close()
