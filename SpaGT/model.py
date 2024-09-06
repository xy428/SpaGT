import torch.nn as nn
from torch.nn import functional as F
from . layer import EGT

class EGT_DEC(nn.Module):
    def __init__(self, model_width=64, edge_width=100, num_heads=5, max_length=None, 
                 gate_attention=True, model_height=4, node_normalization='layer',
                 edge_normalization='layer', l2_reg=0, node_dropout=0, edge_dropout=0, 
                 add_n_norm=False, activation='elu', mlp_layers=[.5, .25], 
                 do_final_norm=True, clip_logits_value=[-5, 5], edge_activation=None, 
                 edge_channel_type='residual', combine_layer_repr=False,
                 ffn_multiplier=2., node2edge_xtalk=0., edge2node_xtalk=0., 
                 global_step_layer=False, scale_degree=False, scaler_type='log', 
                 num_virtual_nodes=0, random_mask_prob=0., attn_dropout=0.,alpha = 0.2):
        super(EGT_DEC, self).__init__()

        if not gate_attention and scale_degree:
            raise ValueError('scale_degree only works with gate_attention')

        self.model_width = model_width
        self.edge_width = edge_width
        self.num_heads = num_heads
        self.max_length = max_length
        self.gate_attention = gate_attention
        self.model_height = model_height
        self.node_normalization = node_normalization
        self.edge_normalization = edge_normalization
        self.add_n_norm = add_n_norm
        self.activation = activation
        self.do_final_norm = do_final_norm
        self.clip_logits_value = clip_logits_value
        self.edge_activation = edge_activation
        self.edge_channel_type = edge_channel_type
        self.combine_layer_repr = combine_layer_repr
        self.ffn_multiplier = ffn_multiplier
        self.node2edge_xtalk = node2edge_xtalk
        self.edge2node_xtalk = edge2node_xtalk
        self.global_step_layer = global_step_layer
        self.scale_degree = scale_degree
        self.scaler_type = scaler_type
        self.num_virtual_nodes = num_virtual_nodes
        self.random_mask_prob = random_mask_prob
        self.attn_dropout = attn_dropout
        self.alpha = alpha

        
        #alpha = float(self.edge_activation[-1])/10

        
        self.h1 = nn.Linear(in_features=self.model_width, out_features=self.model_width*3)
        self.h2 = nn.Linear(in_features=self.model_width*3, out_features=self.model_width)
        self.e1 = nn.Linear(in_features=1, out_features=self.num_heads)
        self.e2 = nn.Linear(in_features=self.num_heads, out_features=1)
        self.r1 = nn.LeakyReLU(negative_slope=alpha)
        self.egt = EGT(num_heads = self.num_heads)
        self.f1 = nn.Linear(in_features=self.model_width, out_features=round(self.model_width * self.ffn_multiplier))
        self.f2 = nn.Linear(in_features=round(self.model_width * self.ffn_multiplier), out_features=self.model_width)
        self.f3 = nn.Linear(in_features=self.edge_width, out_features=round(self.edge_width * self.ffn_multiplier))
        self.f4 = nn.Linear(in_features=round(self.edge_width * self.ffn_multiplier), out_features=self.edge_width)
        #多层mlp
        #self.m1 = nn.Linear(in_features=self.edge_width, out_features=round(f * self.model_width))
        
        norm_dict = {
            'layer': nn.LayerNorm,
            'batch': nn.BatchNorm1d  # 或 BatchNorm2d / BatchNorm3d，取决于数据维度
        }

        # 根据传入的参数选择归一化类型
        self.normlr_node = norm_dict[self.node_normalization](self.model_width)
        self.normlr_edge = norm_dict[self.edge_normalization](self.edge_width)

    def mha_block(self, h, e, gates=None):
        y = h
        
        if not self.add_n_norm:
            h = self.normlr_node(h)
                    
        qkv = self.h1(h)
        h, e, mat = self.egt(qkv ,
                                    ( e         if self.edge_channel_type != 'none' else [] ),
                                    ( gates    if gates is not None                       else [] ))
        
        h = self.h1(h)
        h = self.h2(h)
        # h.dropout
        h = h + y
        if self.add_n_norm:
            h = self.normlr_node(h)
        
        return h,e
    
    def edge_channel(self, e):
        if self.edge_activation is not None:
            # self.edge_activation.lower().startswith('lrelu'):
            e = self.e1(e)
            e = self.r1(e)
        else:
            e = self.e1(e)
            
        return e
    
    def edge_update_none(self, h, e):
        gates = None
        h, _ = self.mha_block( h, e, gates)
        
        return h, e
    
    def edge_up_bias(self, h,e):
        e0 = e
        gates = None
        if self.gate_attention:
            gates = self.e1(e)
        e = self.edge_channel(e)
        h, e = self.mha_block(h, e, gates)
        
        return h, e0

    def edge_up_residual(self, h, e):
        y = e
        if not self.add_n_norm:
            e = self.normlr_edge(e)
        gates = None
        e = e.unsqueeze(-1)
        if self.gate_attention:
            gates = self.e1(e)
        e = self.edge_channel(e)
        h, e = self.mha_block(h, e, gates)
        e = self.e2(e)
        e = e.squeeze(-1)
        #e.dropout
        e = e + y
        if self.add_n_norm:
            e = self.normlr_edge(e) 
        return h, e
    
    def ffnlr1(self, x, flag, normlr):
        y = x
        if not self.add_n_norm:
            x = normlr(x)
        if flag:
            x = self.f1(x)
        else:
            x = self.f3(x)
        return x, y
    
    def ffnlr2(self, x, y, flag, normlr):
        if flag:
            x = self.f2(x)
        else:
            x = self.f4(x)
        #dropout x
        x = x + y
        if not self.add_n_norm:
            x = normlr(x)
        return x
    
    def ffn_block(self, x_h, x_e):
        flag = 1
        x_h, y_h = self.ffnlr1(x_h, flag, self.normlr_node)
        
        if self.edge_channel_type in ['residual', 'constrained']:
            flag = 0
            x_e, y_e = self.ffnlr1(x_e, flag, self.normlr_edge)
            x_e = self.ffnlr2(x_e, y_e, flag, self.normlr_edge)
            
        flag = 1
        x_h = self.ffnlr2(x_h, y_h, flag, self.normlr_node)
        return x_h,x_e

    def forward(self, x, adj):
        h, e = self.edge_up_residual(x, adj)
        h, e = self.ffn_block(h, e)
        return h