import torch.optim
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch, pdb
import matplotlib.pyplot as plt
import argparse, os
import scipy.io
import random


class NeuralNetwork(nn.Module):
    
    def __init__(self): 
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4000, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_network(model,loss_fn,optimizer,train_1, label):
    ep_n = 100
    model.train()
    for ep in tqdm(range(ep_n)):
        for i in range(len(train_1)):
            inp = train_1[i].to(torch.float32)
            inp = inp.unsqueeze(0)
            pred = model(inp)
            inp_l = label[i].unsqueeze(0)
            loss = loss_fn(pred, inp_l)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()       
    
    return model


def test_model(model,test,labels,loss_fn):
    model.eval()
    size = len(test)
    num_batches = len(test)
    test_loss, correct = 0, 0

   
    with torch.no_grad():
        for i in range(size):
            inp_t = test[i].to(torch.float32)
            pred = model(inp_t.unsqueeze(0))
            test_loss += loss_fn(pred, labels[i].unsqueeze(0)).item()
            correct += (pred.argmax(1) == labels[i]).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def generate_trav_waves(speed, T, seg):
    Allu = np.zeros((T, seg))
    Allv = np.zeros((T, seg))
    x = np.arange(0,20,20/seg)
    for i in range(T):
        Allu[i,:] = 0.5+np.sin((np.pi/11)*(x-speed*i))
        Allv[i,:] = 0.5+0.9*np.sin((np.pi/11)*(x-speed*i)+np.pi/10)
    return np.concatenate((Allu,Allv),1)

def generate_stand_waves(period,T,seg):
    Allu = np.zeros((T, seg))
    Allv = np.zeros((T, seg))
    x = np.arange(0,20,20/seg)
    gauss_x = np.exp(-(x-10)**2/60)
    print(gauss_x)
    for i in range(T):
        Allu[i,:] = gauss_x*np.sin(2*1.2*np.pi*i/period)
        Allv[i,:] = 0.2+0.8*gauss_x*np.sin(2*np.pi*i/period)
    return np.concatenate((Allu,Allv),1)

if __name__ == "__main__":
    data_folder = 'FN_train'
    current_directory= os.getcwd()
    directory_path = current_directory+"/FN_train/"
    os.chdir(directory_path)

    FN_params_mat = scipy.io.loadmat('params_train.mat')
    FN_traj_mat = scipy.io.loadmat('data_train.mat')

    FN_params = np.array(FN_params_mat['save_params_train'])
    FN_traj = np.array(FN_traj_mat['data_params_train'])
    
    size_params = FN_params.size
    train = []
    test = []
    labels_train = []
    labels_test = []

    t_dur = 6000
    split = 61
    a = np.linspace(0,t_dur,split)
    print(a)
    for id in range(int(3*size_params/8)):
        for ind in range(1,split):
            tens = torch.from_numpy(FN_traj[id, int(a[ind-1]):int(a[ind]),::20])
            train.append(tens)
            labels_train.append(torch.tensor(0, dtype=torch.long))

    for id in range(int(3*size_params/8),int(size_params/2)):
        for ind in range(1,split):
            tens = torch.from_numpy(FN_traj[id, int(a[ind-1]):int(a[ind]),::20])
            test.append(tens)
            labels_test.append(torch.tensor(0, dtype=torch.long) )

    for id in range(int(size_params/2),int(7*size_params/8)):
        for ind in range(1,split):
            tens = torch.from_numpy(FN_traj[id, int(a[ind-1]):int(a[ind]),::20])
            train.append(tens)
            labels_train.append(torch.tensor(1, dtype=torch.long))

    for id in range(int(7*size_params/8),int(size_params)):
        for ind in range(1,split):
            tens = torch.from_numpy(FN_traj[id, int(a[ind-1]):int(a[ind]),::20])
            test.append(tens)
            labels_test.append(torch.tensor(1, dtype=torch.long) )
    
    c = list(zip(train, labels_train))
    random.shuffle(c)
    train, labels_train = zip(*c)

    
    c = list(zip(test, labels_test))
    random.shuffle(c)
    test, labels_test = zip(*c)

   # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    #print(f"Using {device} device")
    FN_model = NeuralNetwork()
    print(FN_model)
    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(FN_model.parameters(), lr=learning_rate)

    model_trained = train_network(FN_model,loss_fn,optimizer,train,labels_train)

    results = test_model(model_trained, test, labels_test, loss_fn)

    speed_range = np.arange(0.01,0.011,0.0005)
    period_range = np.arange(380,400,20)
    T = 6000
    seg = 400

    test_new = []
    labels_test_new = []
    for speed in speed_range:
        for ind in range(1,split):
            traj = generate_trav_waves(speed,T,seg)
            tens = torch.from_numpy(traj[int(a[ind-1]):int(a[ind]),::20])
            test_new.append(tens)
            labels_test_new.append(torch.tensor(1, dtype=torch.long))

    for period in period_range:
        for ind in range(1,split):
            traj = generate_stand_waves(speed,T,seg)
            tens = torch.from_numpy(traj[int(a[ind-1]):int(a[ind]),::20])
            test_new.append(tens)
            labels_test_new.append(torch.tensor(0, dtype=torch.long) )
    
    
    
    c = list(zip(test_new, labels_test_new))
    random.shuffle(c)
    test_new, labels_test_new = zip(*c)

    results = test_model(model_trained, test_new, labels_test_new, loss_fn)
         




