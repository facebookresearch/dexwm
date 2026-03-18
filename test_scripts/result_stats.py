import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', required=True)
args = parser.parse_args()

folder_name = args.output_dir
all_res = os.listdir(folder_name)
for res_file in all_res:
    res = np.loadtxt(os.path.join(folder_name,res_file))
    if 'pck' in res_file:
        res = res[:,-1]
    print(res_file)
    print('At 4 seconds', res[-1])
    print('Average', np.mean(res,axis=0))
