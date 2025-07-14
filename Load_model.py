# -*- coding: utf-8 -*-

import torch
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader

import sys
sys.path.append('..')

from CustomDataset import CustomDataset
from TransferFPNmodel import TransferNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# In[0] validate Function
def save_validate(val_loader, model, device):
    # switch to evaluate mode
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

# In[2] Parameters setting code 
if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser(description='Transfer Learning for Bathymetry Inversion')
    # Dataset Parameters Setting    
    parser.add_argument('-val_root', metavar='DIR', 
                 default='dataset_results/dataset_test/Hadrah_processed_20240222_test.csv', help='root path of val_domain')
    parser.add_argument('-saved_model', metavar='DIR', 
                        default='saved_best_model/paper_model/Zone_SW22_HI24_Ours.pth')
    # HPO Parameters Setting    
    parser.add_argument('-batch_size', default=1024, type=int, metavar='N',help='mini-batch size (default: 64)') 
    parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--dropout', default=0.20, type=float, metavar='LR')
    parser.add_argument('--trade_off', default=8.90, type=float,help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--trans_loss', default='gram', help='the transfer loss function')
    parser.add_argument('-domain_epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')
    # Model Parameters Setting
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('-model', type=str, default='FFFPyramid',help='Choose the backbone networks')
    parser.add_argument('-hidden_dim', default=16, type=int,help='Dimension of hidden on regressior')
    parser.add_argument('-bottleneck_dim', default=16, type=int,help='Dimension of bottleneck')
    parser.add_argument('-regressor_dim', default=64, type=int,help='Dimension of regressor')
    parser.add_argument('-non-linear', default=False, action='store_true',help='whether not use the linear version')
    parser.add_argument('-input_features', metavar='5',default=14, help='the input feature dims (default is 14).')
    args = parser.parse_args()  
    
    val_dataset = CustomDataset(csv_file=args.val_root, input_dim=args.input_features, use_slope=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=0, drop_last=True)

    model = TransferNet(input_dims=args.input_features-1+6, hidden_dims=args.hidden_dim, 
                        dropout_prob=args.dropout,base_net=args.model, 
                        transfer_loss=args.trans_loss, use_bottleneck=False, 
                        bottleneck_width=args.bottleneck_dim, width=args.regressor_dim).to(device)
    
    model.load_state_dict(torch.load(args.saved_model))
    avg_mse, avg_mae, avg_mape, avg_r2 = save_validate(val_loader, model, device)
    
    print(f'Loading the saved model:')
    print(f'Val_Loss(MSE):{avg_mse:.3f},Val_MAPE:{avg_mape:.3f},Val_R2:{avg_r2:.3f}') 