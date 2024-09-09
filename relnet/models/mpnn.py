import torch 
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as tg
import torch_geometric.nn as tgnn
from torch_scatter import scatter

class MPNNLayer(tgnn.MessagePassing):
    def __init__(self, emb_dim=64, aggr='add'):
        """MPNN Layer"""
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim

        # Message update function
        self.mlp_msg = nn.Sequential(
            nn.Linear(2*emb_dim, emb_dim), 
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), 
            nn.ReLU()
          )
        
        # Node update function
        self.mlp_upd = nn.Sequential(
            nn.Linear(2*emb_dim, emb_dim), 
            nn.ReLU(), 
            nn.Linear(emb_dim, emb_dim), 
            nn.ReLU()
          )
        
    def forward(self, h, edge_index):
        """Forward pass of a message passing layer"""
        out = self.propagate(edge_index, h=h)
        return out
    
    def message(self, h_i, h_j):
        """Compute message for layer update"""
        msg = torch.cat([h_i, h_j], dim=-1)
        return self.mlp_msg(msg)
    
    def aggregate(self, inputs, index):
        """Aggregate messages from neighbors"""
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)
    
    def update(self, aggr_out, h):
        """Update node embeddings"""
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')
    

class InvariantMPNNLayer(tgnn.MessagePassing):
    def __init__(self, emb_dim=64, aggr='add'):
        """MPNN Layer"""
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim

        # Message update function
        self.mlp_msg = nn.Sequential(
            nn.Linear(2*emb_dim + 1, emb_dim), 
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), 
            nn.ReLU()
          )
        
        # Node update function
        self.mlp_upd = nn.Sequential(
            nn.Linear(2*emb_dim, emb_dim), 
            nn.ReLU(), 
            nn.Linear(emb_dim, emb_dim), 
            nn.ReLU()
          )
        
    def forward(self, h, edge_index, dists):
        """Forward pass of a message passing layer"""
        out = self.propagate(edge_index, h=h, dists=dists)
        return out
    
    def message(self, h_i, h_j, dists):
        """Compute message for layer update"""
        msg = torch.cat([h_i, h_j, dists], dim=-1)
        return self.mlp_msg(msg)
    
    def aggregate(self, inputs, index):
        """Aggregate messages from neighbors"""
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)
    
    def update(self, aggr_out, h):
        """Update node embeddings"""
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')
    

class ValueNetwork(nn.Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=7, out_dim=1):
        """Message Passing Neural Network model for graph property prediction"""
        super().__init__()
        
        # Linear projection for node features -> node embeddings
        self.lin_in = nn.Linear(in_dim, emb_dim)
        
        # Message passing layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, aggr='add'))
        
        # Readout function (global mean pool and linear projection)
        self.pool = tgnn.global_mean_pool
        self.lin_pred = nn.Linear(emb_dim, out_dim)
    
    def forward(self, data):
        """Forward pass of the value network"""
        emb = self.lin_in(data.x) # (n, d_n) -> (n, d)
        
        for conv in self.convs:
            emb = emb + conv(emb, data.edge_index)

        if data.batch is None:
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=emb.device)

        # Readout function
        h_graph = self.pool(emb, data.batch)
        out = self.lin_pred(h_graph)

        return out.view(-1)
    

class InvariantValueNetwork(nn.Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=5, out_dim=1):
        """Message Passing Neural Network model for graph property prediction"""
        super().__init__()
        
        # Linear projection for node features -> node embeddings
        self.lin_in = nn.Linear(in_dim, emb_dim)
        
        # Message passing layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(InvariantMPNNLayer(emb_dim, aggr='add'))
        
        # Readout function (global mean pool and linear projection)
        self.pool = tgnn.global_mean_pool
        self.lin_pred = nn.Linear(emb_dim, out_dim)
    
    def forward(self, data):
        """Forward pass of the value network"""
        emb = self.lin_in(data.x) # (n, d_n) -> (n, d)
        
        # Supply inter-node distances to the invariant message passing layers
        for conv in self.convs:
            emb = emb + conv(emb, data.edge_index, data.dists)

        if data.batch is None:
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=emb.device)

        # Readout function
        h_graph = self.pool(emb, data.batch)
        out = self.lin_pred(h_graph)

        return out.view(-1)