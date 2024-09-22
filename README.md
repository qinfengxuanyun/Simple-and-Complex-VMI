# Simple-and-Complex-VMI
## Environment Installation

```bash
# create GPU machine learing environment
conda create -n rapids-24.08 -c rapidsai -c conda-forge -c nvidia  rapids=24.08 python=3.11 'cuda-version>=12.0,<=12.5'
#  install other required packages
pip install -r requirements.txt
```
## Data Preprocessing

```bash
# grating orientation VMI 
python get_eeg_event_data_grating_VMI.py
# object VMI 
python get_eeg_event_data_object_VMI.py
```

## Multiband Spatial Analysis during Visual Mental Imagery

```bash
# grating orientation VMI 
python svm_imagery_ch_sub.py
# object VMI 
python svm_imagery_object_ch_sub.py
```
## Temporal Dynamic Analysis of Visual Mental Imagery within Delta-Theta Bands

```bash
# grating orientation VMI 
python svm_imagery_time-roi_sub.py
python svm_imagery_tgm-roi_sub.py
# object VMI 
python svm_imagery_object_time-roi_sub.py
python svm_imagery_object-roi_sub.py
# plot result
python bio_cls_plot_sub.py
```
## Dynamic Connectivity Analysis of Visual Mental Imagery within Delta-Theta Bands

```bash
# grating orientation VMI 
python bio_imagery_fc_sub.py
# object VMI 
pythonbio_imagery_object_fc_sub.py
```
