# v2.0 
# date   : 12.28
# Author : Bo Zhang
# Content: 
#  - 4 V100 GPU for poc :5 epochs, batch_size = 8 
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
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import customize_data
import modules

def get_args():
    # Get the input parameters
    parser = ArgumentParser(description='ESM-Inpainting model')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature to sample an amino acid')
    parser.add_argument('--noise', type=float, default=0.0, help='Add noise in training')
    parser.add_argument('--lr',type=float,default=1e-2, help="learning rate of Adam optimizer")
    parser.add_argument('--project_name', type=str,default="poc-esm-inpaint",help="project name in wandb")
    parser.add_argument('--data_jsonl', type=str,default="/root/data/chain_set.jsonl",help='Path for the jsonl data')
    parser.add_argument('--split_json', type=str,default="/root/data/splits.json", help='Path for the split json file')
    parser.add_argument('--save_folder',type=str,default="./",help="output folder for the model parameters")
    parser.add_argument('--output_folder',type=str,default="./",help="output folder for the model print information")
    parser.add_argument('--parameter_pattern',type=str,default="min",help="which part of parameters should be frozen")
    parser.add_argument('--parameters',type=str,default="/root/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt", help="parameters path")
    parser.add_argument('--epochs',type=int,default=3,help="epochs to train the model")
    parser.add_argument('--batch_size',type=int,default=6,help="batch size of protein sequences")
    parser.add_argument('--chunk_size',type=int,default=64,help="chunk size of the model")
    parser.add_argument('--max_length',type=int,default=300,help="max length of the dataset")
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--ddp_train", default=True, action='store_true', help="whether to use ddp training model")


    args = parser.parse_args()
    return args

def load_data(args):
    print("start loading data...")
    # data_jsonl = "/root/data/chain_set.jsonl"
    # split_json = "/root/data/splits.json"
    dataset = customize_data.StructureDataset(args.data_jsonl,max_length=args.max_length)
    # Split the dataset
    dataset_indices = {d['name']:i for i,d in enumerate(dataset)}
    with open(args.split_json) as f:
        dataset_splits = json.load(f)
    train_set, validation_set, test_set = [
        Subset(dataset, [
            dataset_indices[chain_name] for chain_name in dataset_splits[key] 
            if chain_name in dataset_indices
        ])
        for key in ['train', 'validation', 'test']
    ]

    print(f"Training:{len(train_set)}, Validation:{len(validation_set)}, Test:{len(test_set)}")
    return train_set, validation_set, test_set


def load_model_optimizer(args,path="/root/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt"):
    # Reading the data file and initialize the esm-inpainting class
    print("start loading model...")
    # path="/root/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt"
    model_path = path
    model_data = torch.load(str(model_path), map_location="cuda:0") #读取一个pickle文件为一个dict
    cfg = model_data["cfg"]["model"]
    model = modules.esm_inpaint(cfg,chunk_size=args.chunk_size,augment_eps=args.noise,pattern=args.parameter_pattern) # make an instance
    model_state = model_data["model"]
    model.esmfold.load_state_dict(model_state, strict=False)
    # model to cuda
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    # define the optimizer
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    return model,optimizer

def initial_wandb(args):
    config = {
        "lr":args.lr,
        "epochs":args.epochs,
        "batch_size":args.batch_size,
        "chunk_size":args.chunk_size,
        "max_length":args.max_length,
        "cuda cores":torch.cuda.device_count(),
        "device name":{i:torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}
    }
    wandb.init(project=args.project_name,config=config)


def main(args):
    train_set, validation_set, test_set = load_data(args)
    loader_train, loader_validation, loader_test = [customize_data.StructureDataloader(
        d, batch_size=args.batch_size
    ) for d in [train_set, validation_set, test_set]]
    model,optimizer = load_model_optimizer(args)
    print("start training...")
    initial_wandb(args)
    # set the best val loss to save the parameters
    best_val_loss = float('inf')
    best_model = None
    best_model_idx = 0
    start_time = time.time()
    device = torch.device("cuda") if torch.cuda.is_availabel() else torch.device("cpu")
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


def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    if not isinstance(tensor,torch.Tensor):
        tensor = torch.as_tensor(tensor)
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def ddp_main(args):
    """
    only training loader
    """
    local_rank = args.local_rank
    # local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')  # nccl的后端通信方式
    device = torch.device("cuda", local_rank)

    # dataset
    print(f"{local_rank} start loading data...")
    # data_jsonl = "/root/data/chain_set.jsonl"
    # split_json = "/root/data/splits.json"
    dataset = customize_data.StructureDataset(
        args.data_jsonl, max_length=args.max_length)
    # Split the dataset
    dataset_indices = {d['name']: i for i, d in enumerate(dataset)}
    with open(args.split_json) as f:
        dataset_splits = json.load(f)
    train_set, validation_set, test_set = [
        Subset(dataset, [
            dataset_indices[chain_name] for chain_name in dataset_splits[key]
            if chain_name in dataset_indices
        ])
        for key in ['train', 'validation', 'test']
    ]

    print(
        f"Training:{len(train_set)}, Validation:{len(validation_set)}, Test:{len(test_set)}")

    # train_set, validation_set, test_set = load_data(args)
    print(f"Rank {local_rank} start making dataloader...")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    loader_train = customize_data.StructureDataloader(
        train_set, batch_size=args.batch_size, shuffle=False, sampler=train_sampler)
    loader_validation, loader_test = [customize_data.StructureDataloader(
        d, batch_size=args.batch_size
    ) for d in [validation_set, test_set]]

    device = torch.device(f"cuda:{local_rank}")
    model_path = args.parameters
    # 读取一个pickle文件为一个dict
    model_data = torch.load(str(model_path), map_location=device)
    cfg = model_data["cfg"]["model"]
    model = modules.esm_inpaint(cfg, chunk_size=args.chunk_size,
                                augment_eps=args.noise, pattern=args.parameter_pattern)  # make an instance
    model_state = model_data["model"]
    model.esmfold.load_state_dict(model_state, strict=False)
    # model to cuda
    model = model.to(device)
    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # DDP方式初始化模型,这种方式会在模型的key上带上"module"
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    optimizer = optim.Adam(ddp_model.parameters(), lr=args.lr)

    if local_rank == 0:
        print("start training...")
        initial_wandb(args)
    # set the best val loss to save the parameters
    best_val_loss = float('inf')
    best_model = None
    best_model_idx = 0
    start_time = time.time()
    dist.barrier() 
    for epoch in range(args.epochs):
        # Training epoch
        train_sampler.set_epoch(epoch)  # 这句莫忘，否则相当于没有shuffle数据
        ddp_model.train()
        total_loss = 0.
        total_loss_seq = 0.
        total_loss_fape = 0.
        log_interval = 200
        for iteration, batch in enumerate(loader_train):
            # move train data to different gpu
            for key in batch:
                batch[key] = batch[key].cuda(local_rank, non_blocking=True)
            optimizer.zero_grad()
            output = ddp_model(coord=batch['mask_coord'], mask=(
                batch['padding_mask']).to(torch.float32), S=batch['mask_seq'])
            # seq loss (nll)
            log_pred = output['log_softmax_aa']
            padding_mask = batch['padding_mask']
            loss_seq_token, loss_seq = utils.loss_nll(
                batch['seq'], log_pred, padding_mask)
            ppl = np.exp(loss_seq.cpu().data.numpy())

            # structure loss (fape)
            B, L = output['positions'].shape[:2]
            pred_position = output['positions'].reshape(B, -1, 3)
            target_position = batch['coord'][:, :, :3, :].reshape(B, -1, 3)
            position_mask = torch.ones_like(target_position[..., 0])
            fape_loss = torch.mean(utils.compute_fape(
                output['pred_frames'], output['target_frames'], batch['padding_mask'], pred_position, target_position, position_mask, 10.0))

            # loss = 0.5*fape + 2*cse copy from DeepMind AlphaFold2 loss function
            loss = 2 * loss_seq + 0.5 * fape_loss
            optimizer.step()
            loss = reduce_mean(loss, dist.get_world_size())
            ppl =  reduce_mean(ppl, dist.get_world_size())
            fape_loss =  reduce_mean(fape_loss, dist.get_world_size())

            # log the loss terms
            if local_rank == 0:
                metric = {"TRAIN/ppl": ppl, 'TRAIN/fape': fape_loss,
                          "TRAIN/ave_loss": loss, "TRAIN/epoch": epoch}
                wandb.log(metric)
                
        # Validation epoch only for rank0, remember validation data loader is the normal sampler
        if local_rank == 0:
            ddp_model.eval()
            with torch.no_grad():
                val_loss = 0
                val_fape = 0
                val_seq = 0
                for _, batch in enumerate(loader_validation):
                    batch = utils.move_batch(batch, device)
                    output = ddp_model(coord=batch['mask_coord'], mask=(
                        batch['padding_mask']).to(torch.float32), S=batch['mask_seq'])
                    # seq loss (nll)
                    log_pred = output['log_softmax_aa']
                    padding_mask = batch['padding_mask']
                    loss_seq_token, loss_seq = utils.loss_nll(
                        batch['seq'], log_pred, padding_mask)

                    # structure loss (fape)
                    B, L = output['positions'].shape[:2]
                    pred_position = output['positions'].reshape(B, -1, 3)
                    target_position = batch['coord'][:,
                                                     :, :3, :].reshape(B, -1, 3)
                    position_mask = torch.ones_like(target_position[..., 0])
                    fape_loss = torch.mean(utils.compute_fape(
                        output['pred_frames'], output['target_frames'], batch['padding_mask'], pred_position, target_position, position_mask, 10.0))
                    # loss = 0.5*fape + 0.2*cse copy from DeepMind AlphaFold2 loss function
                    loss = 2 * loss_seq + 0.5 * fape_loss
                    val_loss += loss.item()
                    val_seq += loss_seq.item()
                    val_fape += fape_loss.item()
                val_loss /= len(loader_validation)
                val_fape /= len(loader_validation)
                val_seq /= len(loader_validation)

                metric = {'VAL/ppl': np.exp(val_seq.cpu().data.numpy()),
                          'VAL/fape': val_fape, "VAL/loss": val_loss}
                wandb.log(metric)
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    best_model = model
                    best_model_idx = epoch
                    param_state_dict = model.module.inpaint_state_dict()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': param_state_dict,
                        'optimizer_state_dict': optimizer.state_dict()
                    }, os.path.join(args.save_folder, f"epoch{epoch}.pt"))
                with open("log.txt", "w") as f:
                    f.write(
                        f"{epoch},{val_loss:.4f},{val_fape:.4f},{np.exp(val_seq.cpu().data.numpy()):.4f}\n")
        dist.barrier() # 这一句作用是：所有进程(gpu)上的代码都执行到这，才会执行该句下面的代码   
    if local_rank == 0:
        # Test epoch
        ddp_model.eval()
        with torch.no_grad():
            test_loss = 0
            test_fape = 0
            test_seq = 0
            for _, batch in enumerate(loader_test):
                batch = utils.move_batch(batch, device)
                output = model(coord=batch['mask_coord'], mask=(
                    batch['padding_mask']).to(torch.float32), S=batch['mask_seq'])
                # seq loss (nll)
                log_pred = output['log_softmax_aa']
                padding_mask = batch['padding_mask']
                loss_seq_token, loss_seq = utils.loss_nll(
                    batch['seq'], log_pred, padding_mask)

                # structure loss (fape)
                B, L = output['positions'].shape[:2]
                pred_position = output['positions'].reshape(B, -1, 3)
                target_position = batch['coord'][:, :, :3, :].reshape(B, -1, 3)
                position_mask = torch.ones_like(target_position[..., 0])
                fape_loss = torch.mean(utils.compute_fape(
                    output['pred_frames'], output['target_frames'], batch['padding_mask'], pred_position, target_position, position_mask, 10.0))
                # loss = 2*fape + 0.2*cse copy from DeepMind AlphaFold2 loss function
                loss = 2 * loss_seq + 0.5 * fape_loss
                test_loss += loss.item()
                test_seq += loss_seq.item()
                test_fape += fape_loss.item()
            test_loss /= len(loader_test)
            test_seq /= len(loader_test)
            test_fape /= len(loader_test)

        print(
            f"Perplexity\tTest{np.exp(test_seq.cpu().data.numpy()) :.4f}\nFape\t{test_fape :.4f}\nLoss\t{test_loss :.4f}")
    dist.barrier() 



if __name__ == "__main__":
    args = get_args()
    if args.ddp_train:
        ddp_main(args)
    else:
        main(args)