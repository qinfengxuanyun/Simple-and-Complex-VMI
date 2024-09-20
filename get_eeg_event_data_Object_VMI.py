import os
import numpy as np
from scipy.io import loadmat
from scipy import signal
import mne
path = "./object_VMI_data/"
original_sfreq = 1000
new_sfreq = 128
F_L = 0.5
F_H = 40
original_order = ['Fp1', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7',
                  'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4',
                  'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6',
                  'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7',
                  'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3',
                  'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2',
                  'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']

info = mne.create_info(
    ch_names=original_order,
    ch_types=['eeg'] * 63,
    sfreq=original_sfreq
)


for sub in os.listdir(path + "pre_data/"):
    data = loadmat(path+ "pre_data/" + sub + "/img.mat")['dataMat']
    print(data.shape[-1])
    for i in range(data.shape[1]):
        os.system(f"mkdir -p {path}event_data_{F_L}-{F_H}_{new_sfreq}/{sub}_{i}/")
        for j in range(12):
            original_data_cut = data[j,i,:,:]
            raw = mne.io.RawArray(original_data_cut, info)
            raw.filter(F_L, F_H)
            # resample to 250 Hz
            raw.resample(new_sfreq)
            data_cut = raw.get_data()
            np.save(f"{path}event_data_{F_L}-{F_H}_{new_sfreq}/{sub}_{i}/{j}.npy", data_cut)
print("finished")