import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import pickle
import random
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
import torch.nn.init as init
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
import thop
import optuna
import plotly
import argparse
import os
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error

def output_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def get_options(
    parser=argparse.ArgumentParser()
):  
    parser.add_argument('--batch_size',
                type=int,
                default=[32,64],
                nargs="+",
                help='Input batch size, default=64')
    parser.add_argument('--epoch',
                type=int,
                default=10,
                help='Number of epochs to train for, default=10')
    parser.add_argument('--lr',
                type=float,
                default=[0.001,0.01],
                nargs="+",
                help='Select the learning rate, default=0.1')
    parser.add_argument('--dropout',
                type=float,
                default=[0.4,0.5],
                nargs="+",
                help='Dropout rate for the first layer, default=0.5')
    parser.add_argument('--input',
                type=str,
                default="test.input",
                help='phen&pca')
    parser.add_argument('--seed',
                type=int,
                default=888,
                help='Random seed, default=888')
    parser.add_argument('--n_trials',
                type=int,
                default=10,
                help='trial time, default=10')

    opt = parser.parse_args()
    if opt.n_trials: 
        print(f'batch_size: {opt.batch_size}')
        print(f'epochs (niters) : {opt.epoch}')
        print(f'learning rate : {opt.lr}')
        print(f'dropout : {opt.dropout}')
        print(f'input : {opt.input}')
        print(f'Random seed : {opt.seed}')
        print(f'trial time : {opt.n_trials}')
    return opt

opt = get_options()
input = opt.input
num_epoch= opt.epoch
min_size, max_size = opt.batch_size[0],opt.batch_size[1]
min_lr,max_lr = opt.lr[0],opt.lr[1]
min_dropout,max_dropout = opt.dropout[0],opt.dropout[1]
seed = opt.seed
n_trials = opt.n_trials
num_feat = 1000
input_val = '/home/weijialin/1.train/data/1_8337_1k/val_bf.1k.input'
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def objective(trial):
    pick=0
    data,num_feat,dataloaders_test,redistribute = prepar(input,trial)
    model = ResNetGS(num_feat,trial).to(device)
    model.custom_weights_init()
    since, test_acc, train_losses, valid_losses,gradient_train,model,epoch_acc,df,mse  = train_model(model, data,dataloaders_test, num_feat = num_feat,redistribute=redistribute,trial=trial)
    print('test_Acc: {:.6f}'.format(test_acc),'Valid_Acc: {:.6f}'.format(epoch_acc),'mse: {:.6f}'.format(mse))
    if test_acc > pick:
        pick=test_acc
        with open("{}.pickle".format(trial.number), "wb") as fout:
            pickle.dump(model, fout)
        rename("{}.pickle".format(trial.number), 'best_trial.pickle')
        df.to_csv("out.txt", index=False,sep='\t')
    param = {
        "batch_size":trial.suggest_int("batch_size", min_size, max_size),
        "lr": trial.suggest_float("lr",min_lr,max_lr),
        "dropout": trial.suggest_float("dropout",min_dropout,max_dropout)
    }
    return test_acc

class ResNetGS(nn.Module):
    def __init__(self,num_feat,trial):
        super(ResNetGS, self).__init__()
        size = trial.suggest_int("batch_size", min_size, max_size)
        dropout = trial.suggest_float("dropout",min_dropout,max_dropout)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=25, stride=1, padding=12, bias=False),
            nn.MaxPool1d(kernel_size=2, stride=2, dilation=1, ceil_mode=False),
            nn.ReLU(inplace=True)
        )
        self.conv_res = nn.Sequential(
            nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv1d(32,64, kernel_size=7, stride=1, padding=3, bias=False)
        )
        self.res_s = nn.Conv1d(32,64, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv_res1 = nn.Sequential(
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv1d(64,128, kernel_size=7, stride=1, padding=3, bias=False)
        )
        self.res_s1 = nn.Conv1d(64,128, kernel_size=1, stride=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_res2 = nn.Sequential(
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv1d(128,256, kernel_size=7, stride=1, padding=3, bias=False)
        )
        self.res_s2 = nn.Conv1d(128,256, kernel_size=1, stride=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.Flatten = nn.Flatten(end_dim=-1)
        self.hidden1 = nn.Sequential(
            nn.Linear(128*num_feat,1024),
            nn.Dropout(p=dropout),
            nn.GELU()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(1024,128),
            nn.Dropout(p=dropout),
        )
        self.out  = nn.Linear(128,1)

    def custom_weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

    def forward(self, x):
        x = self.conv(x)
        r = self.conv_res(x)
        s = self.res_s(x)
        x = r + s
        x = self.relu(x)
        r = self.conv_res1(x)
        s = self.res_s1(x)
        x = r + s
        x = self.relu1(x)
        r = self.conv_res2(x)
        s = self.res_s2(x)
        x = r + s
        x = self.relu2(x)
        x = self.Flatten(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        outputs = self.out(x)
        return outputs

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.   Training on CPU ...')
else:
    print('CUDA is available.   Training on GPU !!!')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, data,dataloaders_test, filename = 'best_model_wts.pth', num_feat = 5,redistribute=0.01,trial=0,num_epoch =num_epoch):
    size = trial.suggest_int("batch_size", min_size, max_size)
    lr = trial.suggest_float("lr", min_lr, max_lr)
    dropout=trial.suggest_float("dropout",min_dropout,max_dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=0.001)
    print('batch_size',size,'lr',lr,'num_epoch',num_epoch,'dropout',dropout)
    criterion = nn.MSELoss()
    since = time.time()
    scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10,verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-10)
    best_loss = np.inf
    epochs_without_improvement = 1
    best_acc = -1
    train_size = int(0.9 * len(data))
    valid_size = len(data) - train_size
    model.to(device)
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    gradient_train = []
    corr = torch.corrcoef
    ad = torch.stack
    for epoch in range(num_epoch):
        llll = []
        train_dataset, valid_dataset = random_split(data, [train_size, valid_size])
        train_dataloader = DataLoader(train_dataset, batch_size= size, shuffle=True,drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size= len(valid_dataset), shuffle=False)
        dataloaders = {'train':train_dataloader,'valid':valid_dataloader}
        print('Epoch {}/{}'.format(epoch+1, num_epoch))
        print('-' * 32)
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = float()
            for inputs,labels in dataloaders[phase]:
                inputs = inputs.unsqueeze(1).float()
                labels = labels.float()
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).float()
                    loss = criterion(outputs, labels).float()
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        tttt = []
                        for param in model.parameters():
                            if param.requires_grad and param.grad is not None:
                                tttt.append(param.grad.sum().item())
                        optimizer.step()
                    else:
                        optimizer.step()
                running_loss += loss.item()
            if phase == 'valid':
                cor = [np.squeeze(outputs, axis=1), labels]
                epoch_acc = corr(torch.stack(cor))[1][0]
            epoch_loss = torch.tensor(running_loss).mean()
            if phase == 'valid':
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                val_loss = epoch_loss.item()
                valid_losses.append(val_loss)
                valid_losses[0] = None
            else:
                train_loss = epoch_loss.item()
                train_losses.append(train_loss)
                train_losses[0] = None
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'valid' and  epoch_acc > best_acc :
                best_acc = epoch_acc
                best_epoch = epoch + 1
                best_model = model
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 1
        else:
            epochs_without_improvement = epochs_without_improvement + 1
        if epochs_without_improvement == 30:
            break
        print()
    print('Best val_Acc: {:.4f}'.format(best_acc),'// train epoch:',num_epoch)
    for inputs,labels in dataloaders_test:
        inputs = inputs.unsqueeze(1).float()
        labels = labels.float()
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.eval()
        outputs = model(inputs).float()
        cor = [np.squeeze(outputs, axis=1), labels]
        test_acc = corr(ad(cor))[1][0]
        real = labels.cpu()
        real1 = real.numpy().tolist()
        pre = np.squeeze(outputs, axis=1)
        pre1 = pre.cpu()
        pre2 = pre1.detach().numpy().tolist()
        df = pd.DataFrame({'obs': real1,'pre': pre2})
        pre_phen = outputs
        print(pre_phen)
        mse = mean_squared_error(pre2,real1)
        gradient_train = 0
    return  since, test_acc,train_losses,valid_losses,gradient_train, model,epoch_acc,df,mse

def test(dataloaders_test,model):
    corr = torch.corrcoef
    ad = torch.stack
    for inputs,labels in dataloaders_test:
        inputs = inputs.unsqueeze(1).float()
        labels = labels.float()
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.eval()
        outputs = model(inputs).float()
        cor = [np.squeeze(outputs, axis=1), labels]
        test_acc = corr(ad(cor))[1][0]
        real = labels.cpu()
        real1 = real.numpy().tolist()
        pre = np.squeeze(outputs, axis=1)
        pre1 = pre.cpu()
        pre2 = pre1.detach().numpy().tolist()
        df = pd.DataFrame({'obs': real1,'pre': pre2})
        pre_phen = outputs
        print(pre_phen)
        mse = mean_squared_error(pre2,real1)
    return test_acc,df,mse

def preData(data):
    PCA = data.iloc[:,1:].values
    num_feat = PCA.shape[1]
    phen = data.iloc[:,0].values
    phen, redistribute = normalized(phen)
    pc = np.empty(shape=PCA.shape)
    for i in range(0,PCA.shape[0]):
        pc[i] = list(PCA[i])
    return pc,phen,num_feat, redistribute

def pretestData(data):
    PCA = data.iloc[:,1:].values
    num_feat = PCA.shape[1]
    phen = data.iloc[:,0].values
    pc = np.empty(shape=PCA.shape)
    for i in range(0,PCA.shape[0]):
        pc[i] = list(PCA[i])
    return pc,phen,num_feat

def prepar(input,trial):
    size = trial.suggest_int("batch_size", min_size, max_size)
    all = pd.read_table(input,sep="\t",header=None)
    data = all.iloc[:int(len(all) * 0.1)]
    test = all.iloc[int(len(all) * 0.1):]
    inputs,labels,num_feat,redistribute = preData(data)
    data = TensorDataset(torch.tensor(inputs), torch.tensor(labels))
    inputs,labels,num_feat = pretestData(test)
    test = TensorDataset(torch.tensor(inputs), torch.tensor(labels))
    dataloaders_test = torch.utils.data.DataLoader(test, batch_size=len(test),num_workers = 64)
    return data, num_feat, dataloaders_test,redistribute

def prepar_test(input):
    all = pd.read_table(input,sep="\t",header=None)
    test = all.iloc[:int(len(all) * 0.1)]
    inputs,labels,num_feat = pretestData(test)
    test = TensorDataset(torch.tensor(inputs), torch.tensor(labels))
    dataloaders_test = torch.utils.data.DataLoader(test, batch_size=len(test))
    return num_feat, dataloaders_test

def test2forward(input):
    all = pd.read_table(input,sep="\t",header=None)
    inputs,labels,num_feat = pretestData(all)
    test = TensorDataset(torch.tensor(inputs), torch.tensor(labels))
    dataloaders_test = torch.utils.data.DataLoader(test, batch_size=len(test))
    return num_feat, dataloaders_test

def normalized(labels):
    mean = np.mean(labels)
    std = np.std(labels)
    normalized_labels = labels / np.max(np.abs(labels))
    redistribute = np.max(np.abs(labels))
    return normalized_labels, redistribute

def rename(src, dst):
    try:
        os.rename(src, dst)
    except FileExistsError:
        os.replace(src, dst)

def detailed_objective(trial):
    data,num_feat,dataloaders_test,redistribute  = prepar(input,trial)
    model = ResNetGS(num_feat,trial).to(device)
    model.custom_weights_init()
    since, test_acc, train_losses, valid_losses,gradient_train,model,epoch_acc,df,mse  = train_model(model, data,dataloaders_test, num_feat = num_feat,redistribute=redistribute,trial=trial)
    print('test_Acc: {:.4f}'.format(test_acc))
    return test_acc

if __name__ == "__main__":
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    with open("best_trial.pickle", 'rb') as file:
        save_model = pickle.load(file).to(device)
        print('study.best_trial.number',study.best_trial.number)
        num_feat,dataloaders_test = prepar_test(input)
        print('save_model',save_model)
        end_acc = test(dataloaders_test, save_model)
        print('end_acc',end_acc)
