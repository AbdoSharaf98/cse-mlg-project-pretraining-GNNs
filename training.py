import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader as pyg_dataloader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from model import GraphLSH

from wilds import get_dataset
from data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper

CRITERION = torch.nn.CrossEntropyLoss()


def add_zeros(data):
        data.x = torch.zeros(data.num_nodes, dtype=torch.long)
        return data

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            save_every: int,
            log_dir: str,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = gpu_id
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.log_dir = log_dir
        self.epochs_run = 0
        
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_checkpoint(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, batch):
        self.optimizer.zero_grad()
        embedding, logits = self.model(batch)
        preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
        is_labeled = batch.y == batch.y
        loss = CRITERION(logits.float(), batch.y.view(-1))
        loss.backward()
        self.optimizer.step()

        return loss.detach(), preds.detach()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        self.train_data.sampler.set_epoch(epoch)
        b = 1
        for batch in self.train_data:
            batch = batch.to(self.gpu_id)
            loss, preds = self._run_batch(batch)
            
            acc = (preds == batch.y.view(-1)).float().mean()
            
            print(f"[GPU{self.gpu_id}] Epoch {epoch} ({b}/{len(self.train_data)}): loss {loss} \t accuracy {acc}", end="\r")
            
            b += 1
         
        print('')

    def _save_checkpoint(self, epoch):
        ckpt = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(ckpt, self.log_dir)
        print(f"Epoch {epoch} | Training checkpoint saved at {self.log_dir}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_molpcba_dataset(root_dir, frac=1.0):

    dataset = get_dataset("ogb-molpcba", download=False, root_dir=root_dir)

    # get the train set and the training loader
    train_data = dataset.get_subset("train", frac=frac)
    test_data = dataset.get_subset("test")

    grouper = CombinatorialGrouper(dataset, ['scaffold'])

    return dataset, train_data, test_data, grouper

def load_ppa_dataset(root_dir):       
    
    dataset = PygGraphPropPredDataset(name='ogbg-ppa', root=root_dir, transform=add_zeros)
    
    split_idx = dataset.get_idx_split()
    
    train_data, valid_data, test_data = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
    
    return dataset, train_data, valid_data, test_data

def prepare_molpcba_dataloader(train_dataset, test_dataset, grouper, train_batch_size, test_batch_size):

    train_loader = get_train_loader(
        "group", train_dataset, grouper=grouper, n_groups_per_batch=8, batch_size=train_batch_size,
        pin_memory=True, shuffle=False, num_workers=4, sampler=DistributedSampler(train_dataset)
    )

    test_loader = get_eval_loader(
        "standard", test_dataset, batch_size=test_batch_size,
        pin_memory=True, shuffle=False, num_workers=4, sampler=DistributedSampler(test_dataset)
    )

    return train_loader, test_loader

def prepare_ppa_dataloader(train_data, valid_data, test_data, train_batch_size, test_batch_size):
    train_loader = pyg_dataloader(train_data,
                                  batch_size=train_batch_size,
                                  pin_memory=True, 
                                  shuffle=False,
                                  sampler=DistributedSampler(train_data))
    valid_loader = pyg_dataloader(valid_data,
                                  batch_size=test_batch_size,
                                  pin_memory=True, 
                                  shuffle=False,
                                  sampler=DistributedSampler(valid_data))
    test_loader = pyg_dataloader(test_data,
                                  batch_size=test_batch_size,
                                  pin_memory=True, 
                                  shuffle=False,
                                  sampler=DistributedSampler(test_data))
    
    return train_loader, valid_loader, test_loader   

def main(rank: int, world_size: int, args):
    
    ddp_setup(rank, world_size)
    
    # data loaders
    train_loader, valid_loader, test_loader = prepare_ppa_dataloader(args.train_data,
                                                                     args.valid_data,
                                                                     args.test_data,
                                                                     args.train_batch_size,
                                                                     args.test_batch_size)

    # model
    model = GraphLSH(args.num_tasks, ppa=True)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    trainer = Trainer(model, train_loader, optimizer, rank, args.save_every, args.log_dir)
    
    trainer.train(args.total_epochs)

    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', type=int, default=50, help='Total epochs to train the model')
    parser.add_argument('--save_every', type=int, default=5, help='How often to save a snapshot')
    parser.add_argument('--train_batch_size', default=64, type=int, help='Input batch size on each device (default: 64)')
    parser.add_argument('--test_batch_size', default=256, type=int)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--root_dir', type=str, default='/storage/ice1/7/2/asharafeldin3/cse8803_project/data')
    parser.add_argument('--log_dir', type=str, default='/storage/ice1/7/2/asharafeldin3/cse8803_project/output')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--frac', type=float, default=1.0)
    main_args = parser.parse_args()
    
    # load the dataset
    dataset, main_args.train_data, main_args.valid_data, main_args.test_data = load_ppa_dataset(root_dir=main_args.root_dir)
    
    main_args.num_tasks = dataset.num_classes
    
    world_size = torch.cuda.device_count()
    
    os.environ["LOCAL_RANK"] = '0'

    mp.spawn(main, args=(world_size, main_args), nprocs=world_size)
