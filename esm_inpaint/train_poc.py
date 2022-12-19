# v1.0 
# date   : 12.19
# Author : Bo Zhang
# Content: 
#  - Single V100 GPU for poc :5 epochs, batch_size = 1 (except multi GPU part)
#  - Training:16631, Validation:1516, Test:1864


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

wandb.init(project="poc-esm-inpaint")
# Get the input parameters
parser = ArgumentParser(description='ESM-Inpainting model')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature to sample an amino acid')
parser.add_argument('--noise', type=float, default=0.1, help='Add noise in training')
parser.add_argument('--data_jsonl', type=str,help='Path for the jsonl data')
parser.add_argument('--split_json', type=str, help='Path for the split json file')
parser.add_argument('--output_folder',type=str,default="output_beta/",help="output folder for the log files and model parameters")
parser.add_argument('--epochs',type=int,default=10,help="epochs to train the model")
parser.add_argument('--batch_size',type=int,default=8,help="batch size of protein sequences")
parser.add_argument('--chunk_size',type=int,default=64,help="chunk size of the model")

args = parser.parse_args()



data_file = "/root/ESM-Inpainting/esm_inpaint/data/chain_set.jsonl"
split_file = "/root/ESM-Inpainting/esm_inpaint/data/splits.json"
dataset = customize_data.StructureDataset(data_file)

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

epoch_losses_train, epoch_losses_valid = [], []
epoch_checkpoints = []
total_step = 0

# Reading the data file and initialize the esm-inpainting class
model_path = "/root/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt"
model_data = torch.load(str(model_path), map_location="cuda:0") #读取一个pickle文件为一个dict
cfg = model_data["cfg"]["model"]
model = modules.esm_inpaint(cfg) # make an instance
model_state = model_data["model"]
model.esmfold.load_state_dict(model_state, strict=False)
model.esmfold.set_chunk_size(args.chunk_size)
model = model.to("cuda")
optimizer = optim.Adam(model.parameters(),lr=1e-3)

for e in range(args.epochs):
    # Training epoch
    model.train()
    total_loss = 0.
    total_loss_seq = 0.
    total_loss_fape = 0.
    log_interval = 200
    for train_i, batch in enumerate(loader_train):
        batch = batch.to("cuda")
        start_batch = time.time()
        # Get a batch
        elapsed_featurize = time.time() - start_batch
        optimizer.zero_grad()
        output = model(coord=batch['mask_coord'],mask=(batch['padding_mask']).to(torch.float32),S=batch['mask_seq'])

        # seq loss (nll)
        log_pred = output['log_softmax_aa']
        padding_mask = batch['padding_mask']
        loss_seq_token, loss_seq = utils.loss_nll(batch['seq'], log_pred, padding_mask)
        ppl = np.exp(loss_av.cpu().data.numpy())


        # structure loss (fape)
        B,L = output['positions'].shape[:2]
        pred_position = output['positions'].reshape(B, -1, 3)
        target_position = batch['coord'][:,:,:3,:].reshape(B, -1 ,3)
        position_mask = torch.ones_like(target_position[...,0])
        fape_loss = torch.sum(utils.compute_fape(output['pred_frames'],output['target_frames'],batch['padding_mask'],pred_position,target_position,position_mask,10.0))

        # loss = 0.5*fape + 0.2*cse copy from DeepMind AlphaFold2 loss function
        loss = 2 * loss_seq + 0.5 * fape_loss
        optimizer.step()
        
        # log the loss terms
        wandb.log({'ppl': ppl, 'fape': fape_loss, "ave_loss":loss})

        total_loss += loss.item()
        total_loss_seq += loss_seq.item()
        total_loss_fape += fape_loss.item()
        if train_i % log_interval == 0 and train_i > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_loss_seq = total_loss_seq / log_interval
            cur_loss_fape = total_loss_fape / log_interval
            cur_loss_ppl = np.exp(cur_loss_seq)
            print(f'| epoch {e:3d} | {train_i:5d}/{len(loader_train):5d} batches | '
                  f'| ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f} | {cur_loss_fape:5.2f}')
            total_loss = 0
            total_loss_seq = 0
            total_loss_fape = 0
            start_time = time.time()

        # Timing
        elapsed_batch = time.time() - start_batch
        total_step += 1
        # print(total_step, elapsed_train, np.exp(loss_av.cpu().data.numpy()), np.exp(loss_av_smoothed.cpu().data.numpy()))
        
        
        # Accumulate true loss
        

        if total_step % 5000 == 0:
            torch.save({
                'epoch': e,
                'model_state_dict': model.inpaint_state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict()
            }, args.output_folder + 'checkpoints/epoch{}_step{}.pt'.format(e+1, total_step))

    # Validation epoch
    model.eval()
    with torch.no_grad():
        validation_sum, validation_weights = 0., 0.
        for _, batch in enumerate(loader_validation):
            X, S, mask, lengths = featurize(batch, device, shuffle_fraction=args.shuffle)
            log_probs = model(X, S, lengths, mask)
            loss, loss_av = loss_nll(S, log_probs, mask)

            # Accumulate
            validation_sum += torch.sum(loss * mask).cpu().data.numpy()
            validation_weights += torch.sum(mask).cpu().data.numpy()

    train_loss = train_sum / train_weights
    train_perplexity = np.exp(train_loss)
    validation_loss = validation_sum / validation_weights
    validation_perplexity = np.exp(validation_loss)
    print('Perplexity\tTrain:{}\t\tValidation:{}'.format(train_perplexity, validation_perplexity))

    # Validation image
    plot_log_probs(log_probs, total_step, folder='{}plots/valid_{}_'.format(base_folder, batch[0]['name']))

    with open(logfile, 'a') as f:
        f.write('{}\t{}\t{}\n'.format(e, train_perplexity, validation_perplexity))

    # Save the model
    checkpoint_filename = base_folder + 'checkpoints/epoch{}_step{}.pt'.format(e+1, total_step)
    torch.save({
        'epoch': e,
        'hyperparams': vars(args),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.optimizer.state_dict()
    }, checkpoint_filename)

    epoch_losses_valid.append(validation_perplexity)
    epoch_losses_train.append(train_perplexity)
    epoch_checkpoints.append(checkpoint_filename)

# Determine best model via early stopping on validation
best_model_idx = np.argmin(epoch_losses_valid).item()
best_checkpoint = epoch_checkpoints[best_model_idx]
train_perplexity = epoch_losses_train[best_model_idx]
validation_perplexity = epoch_losses_valid[best_model_idx]
best_checkpoint_copy = base_folder + 'best_checkpoint_epoch{}.pt'.format(best_model_idx + 1)
shutil.copy(best_checkpoint, best_checkpoint_copy)
load_checkpoint(best_checkpoint_copy, model)


# Test epoch
model.eval()
with torch.no_grad():
    test_sum, test_weights = 0., 0.
    for _, batch in enumerate(loader_test):
        X, S, mask, lengths = featurize(batch, device)
        log_probs = model(X, S, lengths, mask)
        loss, loss_av = loss_nll(S, log_probs, mask)
        # Accumulate
        test_sum += torch.sum(loss * mask).cpu().data.numpy()
        test_weights += torch.sum(mask).cpu().data.numpy()

test_loss = test_sum / test_weights
test_perplexity = np.exp(test_loss)
print('Perplexity\tTest:{}'.format(test_perplexity))

with open(base_folder + 'results.txt', 'w') as f:
    f.write('Best epoch: {}\nPerplexities:\n\tTrain: {}\n\tValidation: {}\n\tTest: {}'.format(
        best_model_idx+1, train_perplexity, validation_perplexity, test_perplexity
    ))