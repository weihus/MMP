
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SubViewEncoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.6):
        super(SubViewEncoder, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        redusial = x
        x = self.gc2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x + redusial
    
class MainViewEncoder(nn.Module):
    def __init__(self, nfeat, nhid):
        super(MainViewEncoder, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)

    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, 0.6, training=self.training)
        # x = self.gc2(x, edge_index)
        # x = F.relu(x)
        return x

class AttributeReconstruction(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.4):
        super(AttributeReconstruction, self).__init__()
        self.gc1 = GCNConv(nhid, nfeat)
        self.gc2 = GCNConv(nfeat, nfeat)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        redusial = x
        x = F.relu(self.gc2(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)

        return x #+ redusial

class StructureReconstruction(nn.Module):
    def __init__(self, nhid, dropout=0.3):
        super(StructureReconstruction, self).__init__()
        self.gc1 = GCNConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, edge_index):
         # x = F.relu(self.gc1(x, edge_index)) 
        # x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T
        x = F.sigmoid(x)
        return x