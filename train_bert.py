
# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from collections import defaultdict
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from dataset import get_dataset

# For data preprocess
import numpy as np
import csv
import os
import time
import datetime

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from src.bert import BertFinetuneModel
from src.metric import compute_metrics
import early_stop as S


def myseed(myseed):
    """set a random seed for reproduciility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def train(train_loader, valid_loader, model, device):
    # set up model
    n_epochs = 200 # Maximum number of epochs
    dict_args = vars(args)
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # loss
    ce_loss_func = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scaler = GradScaler()
    
    start_time = time.time() 
    for epoch in range(1, n_epochs + 1):
        model.train()     
            # for recording training process
        summary_pred = []
        summary_true = []
        losses = []# set model to training mode
        for batch in tqdm(train_loader):    # iterate through the dataloader
            labels = torch.as_tensor(batch['labels'], dtype=torch.float16).to(device)
            optimizer.zero_grad()               # set gradient to zero
            with autocast():
                input_ids = torch.as_tensor(batch['input_ids']).to(device)
                attention_mask = torch.as_tensor(batch['attention_mask']).to(device)
                pred = model(input_ids, attention_mask)
                errCE = ce_loss_func(pred.squeeze(), labels)                  # forward pass (compute output)
            scaler.scale(errCE).backward()
            scaler.step(optimizer)
            scaler.update()
            
            summary_pred.extend(pred.detach().cpu().numpy())
            summary_true.extend(labels.cpu().numpy())
            losses.append(errCE.detach().item())
            
        scheduler.step()
        metrics_dict = compute_metrics(summary_true, summary_pred)
        loss_avg = np.array(losses).mean(axis=0)
        
        print('Training Epoch %d' % epoch,\
                    'CE_loss: %0.4f, MSE: %0.4f, Spearman: %0.4f, MAE: %0.4f, Pearson: %0.4f, R2: %0.4f' % (loss_avg, metrics_dict["mse"], metrics_dict["spearmanr"], metrics_dict["mae"], metrics_dict["pearsonr"], metrics_dict["r2"]))     
        print('Current Learning Rate: %0.6f ' % optimizer.state_dict()['param_groups'][0]['lr'])
           
        eval_loss_epoch = eval(valid_loader, model, device)
        
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
                
        early_stopping = S.EarlyStopping(patience=20)        
        early_stopping(eval_loss_epoch)
        if early_stopping.save_model:
            torch.save(save_file, "save_weights/best_model.pth")

        if early_stopping.early_stop:
            print("We are at epoch:", epoch)
            break
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

        
def eval(valid_loader, model, device):
    model.eval()
    summary_pred = []
    summary_true = []
    losses = []
    
    ce_loss_func = nn.MSELoss()
    
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            labels = torch.as_tensor(batch['labels'], dtype=torch.float16).to(device)
            input_ids = torch.as_tensor(batch['input_ids']).to(device)
            attention_mask = torch.as_tensor(batch['attention_mask']).to(device)
            pred = model(input_ids, attention_mask)
            errCE = ce_loss_func(pred.squeeze(), labels)  
        
            summary_pred.extend(pred.detach().cpu().numpy())
            summary_true.extend(labels.cpu().numpy())
            losses.append(errCE.detach().item())
            
        metrics_dict = compute_metrics(summary_true, summary_pred)
        eval_loss_epoch = np.array(losses).mean(axis=0)
        print('Best Res: MSE: %0.4f, Spearman: %0.4f, MAE: %0.4f, Pearson: %0.4f, R2: %0.4f' % (metrics_dict["mse"], metrics_dict["spearmanr"], metrics_dict["mae"], metrics_dict["pearsonr"], metrics_dict["r2"]))
        
    return eval_loss_epoch 

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Training...")
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument("--bs", type=int, default=10, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0,help="num_workers used in DataLoader")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    args = parse_args()
    
    train_dataset = get_dataset(model_type = "seq", split="train")
    valid_dataset = get_dataset(model_type = "seq", split="valid")
    
    train_loader = DataLoader(train_dataset,batch_size=args.bs,
        shuffle=True,num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset,batch_size=args.bs,
        shuffle=False,num_workers=args.num_workers)
    
    device = get_device()
    
    print('Dataloading Done!')
    print('Train: %d, Tune: %d'%(len(train_dataset), len(valid_dataset)))
    
    dict_args = vars(args)
    print(dict_args)   
    model = BertFinetuneModel().to(device)
        
    train(train_loader, valid_loader, model, device)