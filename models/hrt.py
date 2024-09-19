import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x


class AdaptiveGroupEstimator(nn.Module):
    def __init__(self, function_type=1):
        super().__init__()
        
        self.th = nn.Parameter(torch.Tensor([0.5]))
        self.ste = BinaryThreshold(function_type=function_type)

    def forward(self, node_features):
        temp = F.normalize(node_features, p=2, dim=2)
        corr_mat = torch.matmul(temp, temp.permute(0, 2, 1))
        G = self.ste(corr_mat - self.th.clamp(-0.9999, 0.9999))

        return G


class BinaryThresholdFunctionType1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        # approximation 1
        grad_input[torch.abs(input) > 0.5] = 0

        return grad_input


class BinaryThresholdFunctionType2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        # approximation 2
        grad_input[torch.abs(input) <= 0.5] *= 2 - 4. * torch.abs(input[torch.abs(input) <= 0.5])
        grad_input[torch.abs(input) > 0.5] = 0
        
        return grad_input


class BinaryThresholdFunctionType3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        # approximation 3
        grad_input[torch.abs(input) <= 0.4] *= 2 - 4. * torch.abs(input[torch.abs(input) <= 0.4])
        grad_input[torch.logical_and(torch.abs(input) > 0.4, torch.abs(input) <= 1)] *= 0.4
        grad_input[torch.abs(input) > 1] = 0

        return grad_input


class BinaryThreshold(nn.Module):
    def __init__(self, function_type=1):
        super(BinaryThreshold, self).__init__()
        self.function_type = function_type
        print('[INFO] Binary Threshold Function Type: {}'.format(function_type))

    def forward(self, input):
        if self.function_type == 1:
            return BinaryThresholdFunctionType1.apply(input)
        elif self.function_type == 2:
            return BinaryThresholdFunctionType2.apply(input)
        elif self.function_type == 3:
            return BinaryThresholdFunctionType3.apply(input)
        else:
            raise NotImplementedError


class HRT(nn.Module):
    def __init__(
            self,
            num_layers: int = 3,
            num_heads: int = 4,
            node_dim: int = 64,
            node_hidden_dim: int = 128,
            edge_dim: int = 64,
            edge_hidden_dim_1: int = 128,
            edge_hidden_dim_2: int = 128,
            dropout: float = 0.0,
            aggregation: str = 'avg',
            scale: int = 2,
            function_type: int = 1
    ):
        super(HRT, self).__init__()
        self.scale = scale
        layer = HRTTransformerLayer(
            num_heads,
            node_dim,
            node_hidden_dim,
            edge_dim,
            edge_hidden_dim_1,
            edge_hidden_dim_2,
            dropout,
            aggregation
        )
        
        print('[INFO] HRT Agg: {}'.format(aggregation))
        
        self.aggregation = aggregation # add or att
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.node2edge_mlp = MLP(input_dim=node_dim, output_dim=node_dim, hidden_size=(node_hidden_dim,))
        if self.aggregation == 'att':
            self.attention_mlp = MLP(input_dim=node_dim + edge_dim, output_dim=1, hidden_size=(32,))
        
        if self.scale > 1:
            self.group_gen = AdaptiveGroupEstimator(function_type=function_type)
                
    def init_group_adj(self, node_features):
        G = self.group_gen(node_features)
        
        return G

    def init_pair_adj(self, num_agents, batch_size):
        off_diag = np.ones([num_agents, num_agents])
        
        rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
        rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
        
        G = rel_rec + rel_send
        G = torch.from_numpy(G).type(torch.float).cuda()
        G = G.unsqueeze(0).repeat(batch_size, 1, 1)
        
        return G

    def init_edge(self, x, G):
        N = x.shape[1]
        if self.aggregation == 'avg':
            div = torch.sum(G, dim=-1)[:, :, None]
            edges = torch.matmul(G, x)
            edges = self.node2edge_mlp(edges / div)

        elif self.aggregation == 'sum':
            edges = torch.matmul(G, x)
            edges = self.node2edge_mlp(edges)
            
        elif self.aggregation == 'att':
            x = self.node2edge_mlp(x)
            edge_init = torch.matmul(G, x)
            E = edge_init.shape[1]
            x_rep = (x[:, :, None, :].transpose(2, 1)).repeat(1, E, 1, 1)
            edge_rep = edge_init[:, :, None, :].repeat(1, 1, N, 1)
            node_edge_cat = torch.cat((x_rep, edge_rep), dim=-1)
            attention_weight = self.attention_mlp(node_edge_cat)[:, :, :, 0]
            G_weight = attention_weight * G
            G_weight = F.softmax(G_weight, dim=2)
            G_weight = G_weight * G
            edges = torch.matmul(G_weight, x)
            
        return edges
        
    def forward(self, node_features, _edge_features, _G, return_edge=False):
        if self.scale == 1:
            G = self.init_pair_adj(node_features.shape[1], node_features.shape[0])
        else:
            G = self.init_group_adj(node_features)
        edge_features = self.init_edge(node_features, G)
        
        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, G)
        
        if return_edge:
            return node_features, edge_features, G
        
        return node_features, G


class HRTNoEdgeInit(nn.Module):
    def __init__(
            self,
            num_layers: int = 3,
            num_heads: int = 4,
            node_dim: int = 64,
            node_hidden_dim: int = 128,
            edge_dim: int = 64,
            edge_hidden_dim_1: int = 128,
            edge_hidden_dim_2: int = 128,
            dropout: float = 0.0,
            aggregation: str = 'avg',
            scale: int = 2,
            function_type: int = 1
    ):
        super(HRTNoEdgeInit, self).__init__()
        layer = HRTTransformerLayer(
            num_heads,
            node_dim,
            node_hidden_dim,
            edge_dim,
            edge_hidden_dim_1,
            edge_hidden_dim_2,
            dropout,
            aggregation
        )
        
        self.aggregation = aggregation # add or att
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
                
    def forward(self, node_features, edge_features, G, return_edge=False):
        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, G)
        
        if return_edge:
            return node_features, edge_features, G
            
        return node_features, G


class HRTTransformerLayer(nn.Module):
    def __init__(
        self,
        num_heads: int = 4,
        node_dim: int = 64,
        node_hidden_dim: int = 128,
        edge_dim: int = 64,
        edge_hidden_dim_1: int = 128,
        edge_hidden_dim_2: int = 128,
        dropout: float = 0.0,
        aggregation: str = 'avg'
    ):
        super(HRTTransformerLayer, self).__init__()
        self.aggregation = aggregation
        self.edge_update = True
        
        self.node_attention_layer = HRTAttentionLayer(num_heads, node_dim, edge_dim, aggregation)
            
        self.dropout = nn.Dropout(dropout)
        self.linear_net_n = nn.Sequential(
            nn.Linear(node_dim, node_hidden_dim),
            # nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(node_hidden_dim, node_dim),
        )
        
        self.norm1_n = nn.LayerNorm(node_dim)
        self.norm2_n = nn.LayerNorm(node_dim)
        
        if self.edge_update:
            self.linear_net1_e = nn.Sequential(
                nn.Linear(node_dim + edge_dim, edge_hidden_dim_1),
                # nn.Dropout(dropout),
                nn.ReLU(inplace=True),
                nn.Linear(edge_hidden_dim_1, edge_dim),
            )
            
            self.linear_net2_e = nn.Sequential(
                nn.Linear(edge_dim, edge_hidden_dim_2),
                # nn.Dropout(dropout),
                nn.ReLU(inplace=True),
                nn.Linear(edge_hidden_dim_2, edge_dim),
            )
            
            self.norm1_e = nn.LayerNorm(edge_dim)
            self.norm2_e = nn.LayerNorm(edge_dim)
    
    def forward(self, node_features, edge_features, G):
        node_attn_out = self.node_attention_layer(node_features, edge_features, G)
        node_features = node_features + self.dropout(node_attn_out)
        node_features = self.norm1_n(node_features)
        
        linear_out = self.linear_net_n(node_features)
        node_features = node_features + self.dropout(linear_out)
        node_features = self.norm2_n(node_features)
        
        if self.edge_update:
            if self.aggregation == 'avg':
                div = torch.sum(G, dim=-1)[:, :, None]
                concatenated_inputs = torch.cat((edge_features, torch.matmul(G, node_features) / div), dim=-1)
            elif self.aggregation == 'sum':
                concatenated_inputs = torch.cat((edge_features, torch.matmul(G, node_features)), dim=-1)
            else:
                raise NotImplementedError
            edge_features = edge_features + self.dropout(self.linear_net1_e(concatenated_inputs))
            edge_features = self.norm1_e(edge_features)
            
            edge_features = edge_features + self.dropout(self.linear_net2_e(edge_features))
            edge_features = self.norm2_e(edge_features)
        
        return node_features, edge_features


class HRTAttentionLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        node_dim: int,
        edge_dim: int,
        aggregation: str = 'avg'
    ):
        super(HRTAttentionLayer, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.aggregation = aggregation
        
        self.proj_qkv_n = nn.Linear(node_dim, 3 * node_dim)
        self.proj_qkv_e = nn.Linear(edge_dim, 3 * node_dim)
        self.proj_o = nn.Linear(node_dim, node_dim)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, node_features, edge_features, G):
        B, N, _ = node_features.shape
        B, E, _ = edge_features.shape
        qkv_n = self.proj_qkv_n(node_features)
        qkv_e = self.proj_qkv_e(edge_features)
        
        qkv_n = qkv_n.reshape(B, N, self.num_heads, 3 * self.head_dim)
        qkv_n = qkv_n.permute(0, 2, 1, 3)
        q_n, k_n, v_n = qkv_n.chunk(3, dim=-1)
        
        qkv_e = qkv_e.reshape(B, E, self.num_heads, 3 * self.head_dim)
        qkv_e = qkv_e.permute(0, 2, 1, 3)
        q_e, k_e, v_e = qkv_e.chunk(3, dim=-1)

        if self.aggregation == 'avg':
            div = torch.sum(G, dim=1)[:, None, :, None]
        elif self.aggregation == 'sum':
            div = 1.0
            
        q = q_n + torch.einsum('bmn,bhnc->bhmc', G.transpose(-2, -1), q_e) / div
        k = k_n + torch.einsum('bmn,bhnc->bhmc', G.transpose(-2, -1), k_e) / div
        v = v_n + torch.einsum('bmn,bhnc->bhmc', G.transpose(-2, -1), v_e) / div
        
        qk = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        att_dist = F.softmax(qk, dim=-1)
        
        new_nodes = torch.matmul(att_dist, v) # (B, H, N, D)
        new_nodes = new_nodes.permute(0, 2, 1, 3) # (B, N, H, D)
        new_nodes = new_nodes.reshape(B, N, self.num_heads * self.head_dim) # (B, N, H*D)
        new_nodes = self.proj_o(new_nodes)
        
        return new_nodes
