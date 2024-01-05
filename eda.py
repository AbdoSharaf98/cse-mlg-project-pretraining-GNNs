from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
import torch
import numpy as np

from modules.gnn import GNN
from colorama import Fore

from utils import create_ff_network
from modules.fly_lsh import FlyLSH


# get the dataset
root_dir = "./data"
dataset = get_dataset("ogb-molpcba", download=True, root_dir=root_dir)

# get the train set and the training loader
train_data = dataset.get_subset("train", frac=1.0)
test_data = dataset.get_subset("test")

grouper = CombinatorialGrouper(dataset, ['scaffold'])

# prepare a group-based train loader
train_loader = get_train_loader(
    "group", train_data, grouper=grouper, n_groups_per_batch=8, batch_size=256)
test_loader = get_eval_loader("standard", test_data, batch_size=5000)

# load an example batch
batch, _, batch_metadata = next(iter(train_loader))

# get relevant information about the data set
num_tasks = dataset.ogb_dataset.num_tasks
num_features = dataset.ogb_dataset.num_features
num_edge_features = dataset.ogb_dataset.num_edge_features

""" Build a GNN """
embedding_dim = 300
gnn = GNN(num_tasks, num_layer=5, emb_dim=embedding_dim,
          gnn_type='gin', virtual_node=True, residual=False, drop_ratio=0.5, JK="sum", graph_pooling="mean")

""" Build a FlyLSH layer """
lsh_out_dim = 2000
tag_dim = 6
sr = tag_dim / embedding_dim
fly_lsh = FlyLSH(input_dim=embedding_dim, out_dim=lsh_out_dim, tag_dim=tag_dim, weight_sparsity=sr)
lsh_optimizer = torch.optim.Adam(fly_lsh.parameters(), lr=0.001)

""" Build a downstream classifier """
cls_loss_fn = torch.nn.BCEWithLogitsLoss()
classifier = create_ff_network([embedding_dim, num_tasks], h_activation='none', out_activation="none")
cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

""" Training """
# move model to gpu
device = "cuda"
gnn = gnn.to(device)
classifier = classifier.to(device)
fly_lsh = fly_lsh.to(device)

# create an optimizer
optimizer = torch.optim.Adam(gnn.parameters(), lr=0.003)

num_epochs = 50
num_batches = len(train_loader)
losses = []
cls_losses = []
#accs = []
for epoch in range(num_epochs):
    epoch_losses = []
    epoch_cls_losses = []
    #epoch_accs = []
    epoch_pred = []
    epoch_true = []
    epoch_metadata = []
    batch_itr = 1
    for batch, _, batch_metadata in train_loader:

        # move to device
        batch.to(device)

        # compute embeddings
        _, embedding = gnn(batch)

        # compute lsh tag
        tag = fly_lsh(embedding)

        # pass to classifier
        cls_logits = classifier(embedding)
        cls_pred = (torch.sigmoid(cls_logits) > 0.5).float()

        # compute similarity matching sim_loss
        similarity = (batch_metadata[:, [0]] == batch_metadata[:, [0]].T).float().to(device)
        similarity[similarity == 0] = -1
        embed_sim = ((torch.tanh(embedding) @ torch.tanh(embedding).T) / embedding_dim)
        sim_loss = torch.mean((similarity - embed_sim) ** 2) / 4

        # compute classification sim_loss
        is_labeled = batch.y == batch.y
        cls_loss = cls_loss_fn(cls_logits.float()[is_labeled], batch.y.float()[is_labeled])
        #accuracy = (cls_pred[is_labeled] == batch.y[is_labeled]).float().mean()

        # optimize
        optimizer.zero_grad()
        lsh_optimizer.zero_grad()
        cls_optimizer.zero_grad()

        (cls_loss).backward()

        optimizer.step()
        lsh_optimizer.step()
        cls_optimizer.step()

        # append
        epoch_losses.append(sim_loss.detach().item())
        epoch_cls_losses.append(cls_loss.detach().item())
        #epoch_accs.append(accuracy.detach().item())
        epoch_pred.append(cls_pred)
        epoch_true.append(batch.y)
        epoch_metadata.append(batch_metadata)

        # average precision
        try:
            ap = dataset.eval(cls_pred, batch.y, batch_metadata)[0]['ap']
        except RuntimeError:
            ap = torch.nan

        # update
        print(Fore.YELLOW + f'[train] Epoch {epoch + 1} ({batch_itr} / {num_batches}):\t '
                            f'\033[1mSM Loss\033[0m = {sim_loss.detach().item():0.3f}\t'
                            f'\033[1mBCE Loss\033[0m = {cls_loss.detach().item():0.3f}\t'
                            #f'\033[1mAccuracy\033[0m = {accuracy.detach().item():0.3f}\t'
                            f'\033[1mAP\033[0m = {ap:0.3f}',
              end='\r')

        batch_itr += 1

    print('')

    avg_loss = np.mean(epoch_losses)
    avg_cls_loss = np.mean(epoch_cls_losses)
    #avg_acc = np.mean(epoch_accs)
    epoch_ap = dataset.eval(torch.cat(epoch_pred), torch.cat(epoch_true), torch.cat(epoch_metadata))[0]['ap']
    losses.append(avg_loss)
    cls_losses.append(avg_cls_loss)
    #accs.append(avg_acc)

    if epoch % 1 == 0:
        print(f"Epoch {epoch} | SM Loss {avg_loss: 0.3f} | BCE Loss {avg_cls_loss: 0.3f} | AP {epoch_ap: 0.3f}")


print("Done")


