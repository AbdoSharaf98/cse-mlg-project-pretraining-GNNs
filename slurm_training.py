import os
import builtins
import argparse
import torch
import numpy as np 
import random
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader as pyg_dataloader

from model import GraphLSH

def parse_args():
    parser = argparse.ArgumentParser()
    
    # dataset and model details
    parser.add_argument('--root_dir', type=str, default='/storage/ice1/7/2/asharafeldin3/cse8803_project/data')
    parser.add_argument('--log_dir', type=str, default='/storage/ice1/7/2/asharafeldin3/cse8803_project/output')
    
    # training details
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--train_batch_size', default=64, type=int, help='train batch size per GPU')
    parser.add_argument('--test_batch_size', default=256, type=int, help='test batch size per GPU')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, 
                        help='start epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    
    # DDP configs:
    parser.add_argument('--world-size', default=4, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local-rank', default=-1, type=int, 
                        help='local rank for distributed training')
    args = parser.parse_args()
    return args

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def load_ppa_dataset(root_dir):       

    dataset = PygGraphPropPredDataset(name='ogbg-ppa', root=root_dir)

    split_idx = dataset.get_idx_split()

    train_data, valid_data, test_data = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

    return dataset, train_data, valid_data, test_data

def prepare_ppa_dataloader(train_data, valid_data, test_data, train_batch_size, test_batch_size, n_workers=0):
    train_loader = pyg_dataloader(train_data,
                                  batch_size=train_batch_size,
                                  pin_memory=True, 
                                  shuffle=False,
                                  sampler=DistributedSampler(train_data, shuffle=True),
                                  drop_last=True, num_workers=n_workers)
    valid_loader = pyg_dataloader(valid_data,
                                  batch_size=test_batch_size,
                                  pin_memory=True, 
                                  shuffle=False,
                                  sampler=DistributedSampler(valid_data), drop_last=True, num_workers=n_workers)
    test_loader = pyg_dataloader(test_data,
                                  batch_size=test_batch_size,
                                  pin_memory=True, 
                                  shuffle=False,
                                  sampler=DistributedSampler(test_data), drop_last=True, num_workers=n_workers)
    
    return train_loader, valid_loader, test_loader
                                         
def main(args):
    # DDP setting
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # suppress printing if not on master gpu
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
        
    ### data ###
    dataset, train_data, valid_data, test_data = load_ppa_dataset(args.root_dir)
    train_loader, valid_loader, test_loader = prepare_ppa_dataloader(train_data,
                                                                     valid_data,
                                                                     test_data,
                                                                     args.train_batch_size,
                                                                     args.test_batch_size,
                                                                     args.workers)
       
    ### model ###
    model = GraphLSH(dataset.num_classes, ppa=True)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_without_ddp = model.module
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
        
    ### optimizer ###
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    ### resume training if necessary ###
    if args.resume:
        pass
    
    torch.backends.cudnn.benchmark = True
    
    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)
        # fix sampling seed such that each gpu gets different part of dataset
        if args.distributed: 
            train_loader.sampler.set_epoch(epoch)

        mean_acc, mean_loss = train_one_epoch(train_loader, model, torch.nn.CrossEntropyLoss(), optimizer, epoch, args)
        if args.rank == 0: # only val and save on master node
            train_perf = eval(model, args.rank, train_loader, Evaluator('ogbg-ppa'))
            valid_perf = eval(model, args.rank, valid_loader, Evaluator('ogbg-ppa'))
            test_perf = eval(model, args.rank, test_loader, Evaluator('ogbg-ppa'))
            print(f"Epoch {epoch}: Train {train_perf} \t Valid {valid_perf} \t Test {test_perf}")
            # save checkpoint if needed #

def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args):
    # only one gpu is visible here, so you can send cpu data to gpu by 
    # input_data = input_data.cuda() as normal
    accs, losses = [], []
    b = 1
    for batch in train_loader:
        batch = batch.cuda()
        loss, preds = run_batch(batch, model, optimizer, criterion)
        acc = (preds == batch.y.view(-1)).float().mean()
        accs.append(acc)
        losses.append(loss)
        b += 1 
        print(f"[GPU{args.rank}] Epoch {epoch} ({b}/{len(self.train_loader)}): loss {loss} \t accuracy {acc}", end="\r")
    
    print('')
    
    return np.array(accs).mean(), np.array(losses).mean()

def run_batch(batch, model, optimizer, criterion):
    optimizer.zero_grad()
    embedding, logits = model(batch)
    preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
    loss = criterion(logits.float(), batch.y.view(-1))
    loss.backward()
    optimizer.step()

    return loss.detach(), preds.detach()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                _, pred = model(batch)

            y_true.append(batch.y.view(-1,1).detach().cpu())
            y_pred.append(torch.argmax(torch.softmax(pred.detach(), dim=-1), dim = 1).view(-1,1).cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)
    
def validate(val_loader, model, criterion, epoch, args):
    pass

if __name__ == '__main__':
    args = parse_args()
    main(args)