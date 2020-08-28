import sys
import subprocess
import os

if __name__ == '__main__':
    python_cmd = "python"
    cerebus = False
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        cerebus = True
        python_cmd = "/home/jeremie/GitHub/BoostingBF3S/anaconda3/envs/tensorflow_cuda-env/bin/python3"
        print(os.getcwd())
    else:
        data_dir = r"D:\Datasets\mini-imagenet"

    outfiles_path = "training_data/testing_outputfiles"
    os.makedirs(outfiles_path, exist_ok=True)

    mth_list = ["proto", "proto_rot", "cosine", "cosine_rot", "Gen0", "Gen1"]
    for mth in mth_list:
        with open(f'{outfiles_path}/{mth}_out.txt', 'w') as f:
            process = subprocess.call([python_cmd, 'testing_model.py', data_dir, mth], stdout=f)
            print(process)
