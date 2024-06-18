import torch
import torch.nn.functional as F
import torch.nn as nn
from net.EdgeWeightedGAT import GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_sparse import spspmm
from net.MySAG import SAGPooling


##########################################################################################################################
class Network(torch.nn.Module):
    def __init__(self, indim, ratio, nclass):
        '''

        :param indim: (int) node feature dimension
        :param ratio: (float) pooling ratio in (0,1)
        :param nclass: (int)  number of classes
        :param R: (int) number of ROIs
        '''
        super(Network, self).__init__()

        self.indim = indim
        self.dim1 = 64
        self.dim2 = 64
        self.dim3 = 32
        
        self.conv1 = GATConv(self.indim, self.dim1, dropout = 0.5)
        self.pool1 = SAGPooling(self.dim1, ratio=ratio, GNN=GATConv, nonlinearity=torch.sigmoid) #0.4 data1 10 fold

        self.conv2 = GATConv(self.dim1, self.dim2, dropout = 0.5)
        self.pool2 = SAGPooling(self.dim2, ratio=ratio, GNN=GATConv,nonlinearity=torch.sigmoid)


        # 对phenotype的4列行向量进行FNN约束
        # self.fc0 = torch.nn.Linear(4, 4)
        # self.bn0 = torch.nn.BatchNorm1d(4)

        self.fc1 = torch.nn.Linear((self.dim1+self.dim2)*2 + 12, self.dim2)
        self.bn1 = torch.nn.BatchNorm1d(self.dim2)
        self.fc2 = torch.nn.Linear(self.dim2, self.dim3)
        self.bn2 = torch.nn.BatchNorm1d(self.dim3)
        self.fc3 = torch.nn.Linear(self.dim3, nclass)


    def forward(self, x, edge_index, batch, edge_attr, pos, pen):

        x = self.conv1(x, edge_index,edge_attr)
        x, edge_index, edge_attr, batch, perm, score1 = self.pool1(x, edge_index, edge_attr, batch)

        pos = pos[perm]
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        edge_attr = edge_attr.squeeze()
        edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))

        x = self.conv2(x, edge_index,edge_attr)
        x, edge_index, edge_attr, batch, perm, score2 = self.pool2(x, edge_index,edge_attr, batch)

        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        # pen2 = self.bn0(F.relu(self.fc0(pen)))
        x = torch.cat([x1,x2,pen], dim=1)

        x = self.bn1(F.relu(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.bn2(F.relu(self.fc2(x)))
        x= F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1)

        return x, score1, score2

    
    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

