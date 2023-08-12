import os
cmd = 'pip install gdown'
os.system(cmd)
cmd = 'pip install torch'
os.system(cmd)
import gdown
gdown.download("https://drive.google.com/uc?id=1vh7EeoL4xsOvFf9GdUZQH7S-8Fcb-PJ1")
dir="./data_dir"
if not os.path.exists(dir):
    os.mkdir(dir)
cmd = 'tar -xvzf original_deepsea_dataset_npy.tar.gz -C ./data_dir/'
os.system(cmd)

dir="./model_files"
if not os.path.exists(dir):
    os.mkdir(dir)