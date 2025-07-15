# -*- coding: utf-8 -*-
import math
import torch
import random
import argparse
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna

import sys
sys.path.append('..')
from tllib.utils.data import ForeverDataIterator
from tllib.utils.meter import AverageMeter

from CustomDataset import CustomDataset
from TransferFPNmodel import TransferNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# In[1] validate Function 
def validate(val_loader, model, device):
    model.eval()
    with torch.no_grad():
        features, labels = next(iter(val_loader))
        features = features.to(device).float()
        labels = labels.to(device).float()
        outputs = model.predict(features)
        outputs = outputs.reshape(labels.shape)
        mse = torch.nn.functional.mse_loss(outputs, labels, reduction='mean')
        mae = torch.nn.functional.l1_loss(outputs, labels, reduction='mean')
        mape = torch.mean(torch.abs((labels - outputs) / labels)) * 100
        ss_tot = torch.sum((labels - torch.mean(labels)) ** 2)
        ss_res = torch.sum((labels - outputs) ** 2)
        avg_r2 = 1 - (ss_res / ss_tot)
        
    return mse, mae, mape, avg_r2

# In[2] Train Function 
def train_Transfer(train_source_iter, train_target_iter: ForeverDataIterator, model: TransferNet,
                                       transfer_loss_type, trade_off, iter_per_epochs, optimizer, device):
    MSE_loss = AverageMeter('MSE Loss', ':6.3f')
    T_loss = AverageMeter('Transfer Loss', ':6.3f')
    Total_loss = AverageMeter('Total Loss', ':6.3f')
    model.train()
    for i in range(iter_per_epochs):
        optimizer.zero_grad()
        x_source, labels_source = next(train_source_iter)
        x_source = x_source.to(device).float()
        labels_source = labels_source.to(device).float()
        x_target, labels_target = next(train_target_iter)
        x_target = x_target.to(device).float()
        labels_target = labels_target.to(device).float()        
        labels_source_pred, transfer_loss = model.forward(x_source, x_target, labels_source)
        transfer_loss = transfer_loss*3
        labels_source_pred = labels_source_pred.reshape(labels_source.shape)
        mse_loss = torch.nn.functional.mse_loss(labels_source_pred, labels_source)
        loss = mse_loss + transfer_loss * trade_off
        if transfer_loss_type == 'none':
            T_loss.update(0, x_source.size(0))
        else:
            T_loss.update(transfer_loss.item(), x_source.size(0))
        MSE_loss.update(mse_loss.item(), x_source.size(0))
        Total_loss.update(loss.item(), x_source.size(0))

        loss.backward()
        optimizer.step()
        
    return Total_loss.avg, MSE_loss.avg, T_loss.avg
# In[3] Train——Main Function
def Main_train_Transfer(args):
    # Data loading code 
    train_source_dataset = CustomDataset(csv_file=args.train_source_root, input_dim=args.input_features, use_slope=True)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True,pin_memory=True)
     
    train_target_dataset = CustomDataset(csv_file=args.train_target_root, input_dim=args.input_features, use_slope=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True,pin_memory=True)
     
    val_dataset = CustomDataset(csv_file=args.val_root, input_dim=args.input_features, use_slope=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=0, drop_last=True)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)
    min_length = min(len(train_source_dataset), len(train_target_dataset)) 

    model = TransferNet(input_dims=args.input_features-1+6, hidden_dims=args.hidden_dim, 
                        dropout_prob=args.dropout,
                        base_net=args.model, transfer_loss=args.trans_loss, use_bottleneck=False, 
                        bottleneck_width=args.bottleneck_dim, width=args.regressor_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float('inf')
    stop = 0
    iter_per_epochs = math.ceil(min_length/args.batch_size) 
    for epoch in range(args.domain_epochs):    
        stop += 1
        Loss, regress_loss, dann_loss = train_Transfer(train_source_iter, train_target_iter, 
                                                       model, args.trans_loss, args.trade_off, 
                                                       iter_per_epochs, optimizer, device)
        print(f'Epoch [{epoch + 1}/{args.domain_epochs}]-T_Loss:{Loss:.4f},R_loss:{regress_loss:.3f},D_loss:{dann_loss:.3f}')
        avg_mse, avg_mae, avg_mape, avg_r2 = validate(val_loader, model, device)
        print(f'Val_Loss(MSE):{avg_mse:.3f},Val_MAPE:{avg_mape:.3f},Val_R2:{avg_r2:.3f}')
        if avg_mse < best_loss:
            best_loss = avg_mse
            torch.save(model.state_dict(), 'best_model.pth')
            stop = 0
        if stop >= args.early_stop:
            break
        
# In[4] Optuna for HPO
def objective(trial):
    global best_loss
    random.seed(SEED)
    params_search = {
        "batch_size": trial.suggest_categorical('batch_size', [512, 1024]), 
        "lr": trial.suggest_loguniform('lr', 1e-4, 1e-1), 
        "dropout": trial.suggest_uniform('dropout', 0.1, 0.7),
        "trade_off": trial.suggest_uniform('trade_off', 0.3, 30)
            }
    # Data loading code 
    train_source_dataset = CustomDataset(csv_file=args.train_source_root, input_dim=args.input_features, use_slope=True)
    train_source_loader = DataLoader(train_source_dataset, batch_size=params_search["batch_size"], shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
    
    train_target_dataset = CustomDataset(csv_file=args.train_target_root, input_dim=args.input_features, use_slope=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=params_search["batch_size"], shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
    
    val_dataset = CustomDataset(csv_file=args.val_root, input_dim=args.input_features, use_slope=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=args.workers, drop_last=True)
    
    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)
    min_length = min(len(train_source_dataset), len(train_target_dataset))

    model = TransferNet(input_dims=args.input_features-1+6, hidden_dims=args.hidden_dim, 
                        dropout_prob=params_search["dropout"],
                        base_net=args.model, transfer_loss=args.trans_loss, use_bottleneck=False, 
                        bottleneck_width=args.bottleneck_dim, width=args.regressor_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params_search["lr"])
    best_loss_local = float('inf')
    stop = 0
    iter_per_epochs = math.ceil(min_length / params_search["batch_size"]) 
    for epoch in range(args.domain_epochs):
        stop += 1
        Loss, regress_loss, dann_loss = train_Transfer(train_source_iter, train_target_iter, 
                                                           model, args.trans_loss, params_search["trade_off"], 
                                                           iter_per_epochs, optimizer, device)
        print(f'Epoch [{epoch + 1}/{args.domain_epochs}]-T_Loss:{Loss:.4f},R_loss:{regress_loss:.3f},D_loss:{dann_loss:.3f}')
        avg_mse, avg_mae, avg_mape, avg_r2 = validate(val_loader, model, device)
        print(f'Val_Loss(MSE):{avg_mse:.3f},Val_MAPE:{avg_mape:.3f},Val_R2:{avg_r2:.3f}')

        if avg_mse < best_loss_local:
            best_loss_local = avg_mse
            stop = 0
            if best_loss_local < best_loss:
                best_loss = best_loss_local
                torch.save(model.state_dict(), 'best_model.pth')
        if stop >= args.early_stop:
            break
    return best_loss_local
if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser(description='Transfer Learning for Bathymetry Inversion')    
    parser.add_argument('-train_source_root', metavar='DIR',
                        default='dataset_results/dataset_train/Bimini_processed_20200101_train.csv', help='root path of train_source_domain')
    parser.add_argument('-train_target_root', metavar='DIR',
                        default='dataset_results/dataset_train/Bimini_processed_20230301_train.csv', help='root path of train_target_domain')
    parser.add_argument('-val_root', metavar='DIR',
                        default='dataset_results/dataset_test/Bimini_processed_20230301_test.csv', help='root path of val_domain')
    parser.add_argument('-input_features', metavar='5',default=14, help='the input feature dims (default is 14).')
    parser.add_argument('-batch_size', default=1024, type=int, metavar='N',help='mini-batch size (default: 64)') 
    parser.add_argument('--lr', default=0.003, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--dropout', default=0.2, type=float, metavar='LR')
    parser.add_argument('--trade_off', default=16.26, type=float,help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--trans_loss', default='gram', help='the transfer loss function')
    parser.add_argument('-domain_epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')
    # Transfer Parameters Setting
    parser.add_argument('--early_stop', type=int, default=20)
    # Model Parameters Setting
    parser.add_argument('-model', type=str, default='FFFPyramid',help='Choose the backbone networks')
    parser.add_argument('-hidden_dim', default=16, type=int,help='Dimension of hidden on regressior')
    parser.add_argument('-bottleneck_dim', default=16, type=int,help='Dimension of bottleneck')
    parser.add_argument('-regressor_dim', default=64, type=int,help='Dimension of regressor')
    parser.add_argument('-non-linear', default=False, action='store_true',help='whether not use the linear version')
    # Log Model Parameters
    args = parser.parse_args()    
    # In[6] Train and Val using Optuna 
    best_loss = float('inf')
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10) 
    best_params = study.best_params
    # In[7] Train and Val using Main_train_Transfer
    # Main_train_Transfer(args)
