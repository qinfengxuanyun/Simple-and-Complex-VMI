import pandas as pd
import numpy as np
import os
import mne
original_order = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3',
                 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1',
                 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz',
                 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2',
                 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1',
                 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz',
                 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6',
                 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2', 'Iz']

original_sfreq = 1024
new_sfreq = 128
ratio = original_sfreq // new_sfreq
info = mne.create_info(
    ch_names=original_order,
    ch_types=['eeg'] * 64,
    sfreq=original_sfreq
)
event_code_dict = {1: 0, 2: 1}
index_dict = {1:0, 2:1}
duration = 6
F_L = 30
F_H = 40
path = "./grating_VMI_data/"
input_dir = path+"ica_ref_notch_0.53-40/"
output_dir = path+f"event_data_{F_L}-{F_H}_{duration}s_{new_sfreq}_bs/"
os.system("mkdir -p {}".format(output_dir))

stas_list = []
cls1_num = 0
cls0_num = 0
for file in os.listdir(input_dir):
    sub = file.split('-')[0]
    if file.split(".")[-1] == "npy" or  file.split(".")[0][-4:] == "VVIQ":
        continue
    print(file)
    sub_dir = output_dir + file.split(".")[0]
    os.system("mkdir -p {}".format(sub_dir))
    events = pd.read_csv(input_dir+file).values.tolist()
    data = np.load(input_dir+file.replace('csv','npy'))
    raw = mne.io.RawArray(data, info)
    raw.filter(l_freq=F_L, h_freq=F_H)
    # resample to 128Hz
    raw.resample(new_sfreq)
    data = raw.get_data()

    bad_num = [0 for i in range(2)]
    vaild_num = [0 for i in range(2)]
    for index, event in enumerate(events):
        if event_code_dict[event[2]]:
            cls1_num += 1
        else:
            cls0_num += 1
        data_cut = data[:,event[0]//ratio-int(0.2*new_sfreq):event[0]//ratio+duration*new_sfreq]

        data_cut_min = np.min(data_cut)
        data_cut_max = np.max(data_cut)
        #100uV
        if data_cut_min < -1e-4 or data_cut_max > 1e-4:
            bad_num[index_dict[event[2]]] += 1
        else:
            vaild_num[index_dict[event[2]]] += 1
        np.save(sub_dir + "/" + str(event[2]) + "_" + str(index) + ".npy", data_cut)
    stas_now = [file.split(".")[0]] + vaild_num + bad_num
    stas_list.append(stas_now)
df = pd.DataFrame(stas_list, columns=['file'] + [str(i) for i in range(2)] + ['b' + str(i) for i in range(2)])
os.system("touch {}".format(path + "vaild_stat.csv"))
df.to_csv(path + "valid_stat.csv", sep=',', index=False)

##########################################################################
data = pd.read_csv(path + "valid_stat.csv",parse_dates=False).values[:,1:]
data_sum = np.sum(data,axis=0)
print(data_sum)
