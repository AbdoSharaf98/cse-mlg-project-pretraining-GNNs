import torch
from torch import nn
from modules.gnn import GNN
from modules.gnn_ppa import GNN as GNN_ppa
from modules.fly_lsh import FlyLSH
from utils import create_ff_network


class GraphLSH(nn.Module):

    def __init__(self, num_tasks,
                 num_gnn_layers=5,
                 embedding_dim=300,
                 lsh_out_dim=2000,
                 lsh_tag_dim=6,
                ppa=False):
        super().__init__()

        """ Build a GNN """
        if not ppa:
            self.gnn = GNN(num_tasks, num_layer=num_gnn_layers, emb_dim=embedding_dim,
                           gnn_type='gin', virtual_node=True, residual=False, drop_ratio=0.5, JK="sum",
                           graph_pooling="mean")
        else:
            self.gnn = GNN_ppa(num_tasks, num_layer=num_gnn_layers, emb_dim=embedding_dim,
                           gnn_type='gin', virtual_node=True, residual=False, drop_ratio=0.5, JK="sum",
                           graph_pooling="mean")    

        """ Build a FlyLSH layer """
        #sr = lsh_tag_dim / embedding_dim
        #self.fly_lsh = FlyLSH(input_dim=embedding_dim, out_dim=lsh_out_dim, tag_dim=lsh_tag_dim, weight_sparsity=sr)

        """ Build a downstream classifier """
        #self.classifier = create_ff_network([embedding_dim, num_tasks], h_activation='none', out_activation="none")

    def forward(self, batch):
        # compute embeddings
        logits, embedding = self.gnn(batch)

        # compute lsh tag
        #tag = self.fly_lsh(embedding)

        # pass to classifier
        #logits = self.classifier(embedding)  # TODO

        return embedding, logits
