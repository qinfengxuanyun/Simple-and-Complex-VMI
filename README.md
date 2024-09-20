# Simple-and-Complex-VMI
## Environment Installation

```bash
# create GPU machine learing environment
conda create -n rapids-24.08 -c rapidsai -c conda-forge -c nvidia  rapids=24.08 python=3.11 'cuda-version>=12.0,<=12.5'
#  install other required packages
pip install -r requirements.txt
```
