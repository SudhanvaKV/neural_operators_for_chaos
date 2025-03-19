import torch
import numpy as np
import pdb, random
import argparse, os
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import scipy.io

############################### validate the training data######################
data_folder, random_index = 'FN_train', 0
current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")
directory_path = current_directory+ "/FN_train/"
os.chdir(directory_path)

FN_params_train = scipy.io.loadmat('params_train.mat')
FN_traj_train = scipy.io.loadmat('data_train.mat')

FN_params_train = np.array(FN_params_train['save_params_train'])
FN_traj = np.array(FN_traj_train['data_params_train'])
FN_traj = FN_traj[:,::20,::20]
print(FN_traj.shape)

torch.save(FN_params_train, f'{directory_path}/training_params.pth')

for ix in range(FN_traj.shape[0]):
    torch.save({'0': FN_params_train[ix], '1': FN_traj[ix]}, '{}/{:06d}.pth'.format(directory_path,ix))
    
directory_path = current_directory+ "/FN_val/"
os.chdir(directory_path)


torch.save(FN_params_train, f'{directory_path}/training_params.pth')

for ix in range(FN_traj.shape[0]):
    torch.save({'0': FN_params_train[ix], '1': FN_traj[ix]}, '{}/{:06d}.pth'.format(directory_path,ix))
    
directory_path = current_directory+ "/FN_test/"
os.chdir(directory_path)


torch.save(FN_params_train, f'{directory_path}/training_params.pth')

for ix in range(FN_traj.shape[0]):
    torch.save({'0': FN_params_train[ix], '1': FN_traj[ix]}, '{}/{:06d}.pth'.format(directory_path,ix))
