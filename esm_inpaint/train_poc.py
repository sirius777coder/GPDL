# v1.0 
# date   : 12.19
# Author : Bo Zhang
# Content: 
#  - Single V100 GPU for poc :5 epochs, batch_size = 8 
#  - Training:16631, Validation:1516, Test:1864
# Visualization : wanb

import json,time,os,sys,shutil,wandb
import utils
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Subset

import customize_data
import modules

# Get the input parameters
parser = ArgumentParser(description='ESM-Inpainting model')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature to sample an amino acid')
parser.add_argument('--noise', type=float, default=0.1, help='Add noise in training')
parser.add_argument('--lr',type=float,default=1e-3, help="learning rate of Adam optimizer")
parser.add_argument('--project_name', type=str,default="poc-esm-inpaint",help="project name in wandb")
parser.add_argument('--data_jsonl', type=str,help='Path for the jsonl data')
parser.add_argument('--split_json', type=str, help='Path for the split json file')
parser.add_argument('--save_folder',type=str,default="./",help="output folder for the model parameters")
parser.add_argument('--output_folder',type=str,default="./",help="output folder for the model print information")
parser.add_argument('--epochs',type=int,default=2,help="epochs to train the model")
parser.add_argument('--batch_size',type=int,default=3,help="batch size of protein sequences")
parser.add_argument('--chunk_size',type=int,default=64,help="chunk size of the model")
parser.add_argument('--max_length',type=int,default=300,help="max length of the dataset")


args = parser.parse_args()


print("start loading data...")
data_file = "/root/ESM-Inpainting/esm_inpaint/data/chain_set.jsonl"
split_file = "/root/ESM-Inpainting/esm_inpaint/data/splits.json"
dataset = customize_data.StructureDataset(data_file,max_length=args.max_length)

# Split the dataset
dataset_indices = {d['name']:i for i,d in enumerate(dataset)}
with open(split_file) as f:
    dataset_splits = json.load(f)
train_set, validation_set, test_set = [
    Subset(dataset, [
        dataset_indices[chain_name] for chain_name in dataset_splits[key] 
        if chain_name in dataset_indices
    ])
    for key in ['train', 'validation', 'test']
]
loader_train, loader_validation, loader_test = [customize_data.StructureDataloader(
    d, batch_size=args.batch_size
) for d in [train_set, validation_set, test_set]]
print(f"Training:{len(train_set)}, Validation:{len(validation_set)}, Test:{len(test_set)}")

config = {
    "lr":args.lr,
    "epochs":args.epochs,
    "batch_size":args.batch_size,
    "chunk_size":args.chunk_size,
    "max_length":args.max_length,
    "len_trainset":len(train_set),
    "len_valset":len(validation_set),
    "len_testset":len(test_set),
    "cuda cores":torch.cuda.device_count(),
    "device name":{i:torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}
}


# Reading the data file and initialize the esm-inpainting class
print("start loading model...")
model_path = "/root/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt"
model_data = torch.load(str(model_path), map_location="cuda:0") #读取一个pickle文件为一个dict
cfg = model_data["cfg"]["model"]
model = modules.esm_inpaint(cfg) # make an instance
model_state = model_data["model"]
model.esmfold.load_state_dict(model_state, strict=False)
model.esmfold.set_chunk_size(args.chunk_size)
# model to cuda
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
# define the optimizer
optimizer = optim.Adam(model.parameters(),lr=args.lr)

# set the best val loss to save the parameters
best_val_loss = float('inf')
best_model = None
best_model_idx = 0

wandb.init(project=args.project_name,config=config)
print("start training...")
start_time = time.time()

for epoch in range(args.epochs):
    # Training epoch
    model.train()
    total_loss = 0.
    total_loss_seq = 0.
    total_loss_fape = 0.
    log_interval = 200
    for iteration, batch in enumerate(loader_train):
        batch = utils.move_batch(batch,device)
        start_batch = time.time()
        optimizer.zero_grad()
        output = model(coord=batch['mask_coord'],mask=(batch['padding_mask']).to(torch.float32),S=batch['mask_seq'])
        # seq loss (nll)
        log_pred = output['log_softmax_aa']
        padding_mask = batch['padding_mask']
        loss_seq_token, loss_seq = utils.loss_nll(batch['seq'], log_pred, padding_mask)
        ppl = np.exp(loss_seq.cpu().data.numpy())


        # structure loss (fape)
        B,L = output['positions'].shape[:2]
        pred_position = output['positions'].reshape(B, -1, 3)
        target_position = batch['coord'][:,:,:3,:].reshape(B, -1 ,3)
        position_mask = torch.ones_like(target_position[...,0])
        fape_loss = torch.mean(utils.compute_fape(output['pred_frames'],output['target_frames'],batch['padding_mask'],pred_position,target_position,position_mask,10.0))

        # loss = 0.5*fape + 2*cse copy from DeepMind AlphaFold2 loss function
        loss = 2 * loss_seq + 0.5 * fape_loss
        optimizer.step()
        
        # log the loss terms
        metric = {"TRAIN/ppl":ppl,'TRAIN/fape': fape_loss, "TRAIN/ave_loss":loss, "TRAIN/epoch":epoch}
        wandb.log(metric)
        total_loss += loss.item()
        total_loss_seq += loss_seq.item()
        total_loss_fape += fape_loss.item()
        if iteration % log_interval == 0 and iteration > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_loss_seq = total_loss_seq / log_interval
            cur_loss_fape = total_loss_fape / log_interval
            cur_loss_ppl = np.exp(cur_loss_seq)
            print(f'| epoch {epoch:3d} | {iteration:5d}/{len(loader_train):5d} batches | '
                  f'| ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {cur_loss_ppl:5.2f} | {cur_loss_fape:5.2f}')
            total_loss = 0
            total_loss_seq = 0
            total_loss_fape = 0
            start_time = time.time()                
    # Validation epoch
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_fape = 0 
        val_seq = 0  
        for _, batch in enumerate(loader_validation):
            batch = utils.move_batch(batch,device)
            output = model(coord=batch['mask_coord'],mask=(batch['padding_mask']).to(torch.float32),S=batch['mask_seq'])
            # seq loss (nll)
            log_pred = output['log_softmax_aa']
            padding_mask = batch['padding_mask']
            loss_seq_token, loss_seq = utils.loss_nll(batch['seq'], log_pred, padding_mask)

            # structure loss (fape)
            B,L = output['positions'].shape[:2]
            pred_position = output['positions'].reshape(B, -1, 3)
            target_position = batch['coord'][:,:,:3,:].reshape(B, -1 ,3)
            position_mask = torch.ones_like(target_position[...,0])
            fape_loss = torch.mean(utils.compute_fape(output['pred_frames'],output['target_frames'],batch['padding_mask'],pred_position,target_position,position_mask,10.0))
            # loss = 0.5*fape + 0.2*cse copy from DeepMind AlphaFold2 loss function
            loss = 2 * loss_seq + 0.5 * fape_loss            
            val_loss += loss.item()
            val_seq += loss_seq.item()
            val_fape += fape_loss.item()
        val_loss /= len(loader_validation)
        val_fape /= len(loader_validation)
        val_seq /= len(loader_validation)

        metric = {'VAL/ppl': np.exp(val_seq.cpu().data.numpy()), 'VAL/fape': val_fape, "VAL/loss":val_loss}
        wandb.log(metric)
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            best_model = model
            best_model_idx = epoch
            param_state_dict = model.inpaint_state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': param_state_dict,
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(args.save_folder,f"epoch{epoch}.pt"))
        with open("log.txt","w") as f:
            f.write(f"{epoch},{val_loss:.4f},{val_fape:.4f},{np.exp(val_seq.cpu().data.numpy()):.4f}\n")

# Test epoch
model.eval()
with torch.no_grad():
    test_loss = 0
    test_fape = 0 
    test_seq = 0  
    for _, batch in enumerate(loader_test):
        batch = utils.move_batch(batch,device)
        output = model(coord=batch['mask_coord'],mask=(batch['padding_mask']).to(torch.float32),S=batch['mask_seq'])
        # seq loss (nll)
        log_pred = output['log_softmax_aa']
        padding_mask = batch['padding_mask']
        loss_seq_token, loss_seq = utils.loss_nll(batch['seq'], log_pred, padding_mask)

        # structure loss (fape)
        B,L = output['positions'].shape[:2]
        pred_position = output['positions'].reshape(B, -1, 3)
        target_position = batch['coord'][:,:,:3,:].reshape(B, -1 ,3)
        position_mask = torch.ones_like(target_position[...,0])
        fape_loss = torch.mean(utils.compute_fape(output['pred_frames'],output['target_frames'],batch['padding_mask'],pred_position,target_position,position_mask,10.0))
        # loss = 2*fape + 0.2*cse copy from DeepMind AlphaFold2 loss function
        loss =2 * loss_seq + 0.5 * fape_loss            
        test_loss += loss.item()
        test_seq += loss_seq.item()
        test_fape += fape_loss.item()
    test_loss /= len(loader_test)
    test_seq /= len(loader_test)
    test_fape /= len(loader_test)

print(f"Perplexity\tTest{np.exp(test_seq.cpu().data.numpy()) :.4f}\nFape\t{test_fape :.4f}\nLoss\t{test_loss :.4f}")