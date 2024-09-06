import torch
import torch.nn as nn
import torch.nn.functional as F

class  EGT(nn.Module):
    def __init__(self,
                 num_heads=2,
                 clip_logits_value=[-5, 5],
                 scale_degree=False,
                 scaler_type='log',
                 edge_input=True,
                 gate_input=True,
                 attn_mask=False,
                 num_virtual_nodes=0,
                 random_mask_prob=0.0,
                 attn_dropout=0.2):
        super(EGT, self).__init__()
        
        self.supports_masking = True
        self.num_heads = num_heads
        self.clip_logits_value = clip_logits_value
        self.scale_degree = scale_degree
        self.scaler_type = scaler_type
        self.edge_input = edge_input
        self.gate_input = gate_input
        self.attn_mask = attn_mask
        self.num_virtual_nodes = num_virtual_nodes
        self.random_mask_prob = random_mask_prob
        self.attn_dropout = attn_dropout

        if scale_degree and not gate_input:
            raise ValueError('scale_degree requires gate_input')

        if scaler_type not in ('log', 'linear'):
            raise ValueError('scaler_type must be log or linear')

    def forward(self, QKV, E,G, mask=None):
        # if mask is None:
        #     mask = torch.ones(len(inputs), len(inputs)).bool()
        # else:
        #     mask = mask[0]

        # # get inputs
        # # QKV, *inputs = inputs  # l,3dh
        # # if self.edge_input:
        # #     E, *inputs = inputs  # l,l,h
        # # G, *inputs = inputs  # l,l,h
        # if self.attn_mask:
        #     M, *inputs = inputs  # l,l,h

        # split query, key, value
        QKV_shape = QKV.size()
        
        assert QKV_shape[1] % (self.num_heads * 3) == 0
        dot_dim = QKV_shape[1] // (self.num_heads * 3)
        QKV = QKV.view(QKV_shape[0], 3, dot_dim, self.num_heads)  # l,3dh -> l,3,d,h
        Q, K, V = QKV.unbind(dim=1)  # l,d,h

        # form attention logits from nodes
        A_hat = torch.einsum('ldh,mdh->lmh', Q, K) * (dot_dim ** -0.5)  # l,l,h
        if self.clip_logits_value is not None:
            A_hat = torch.clamp(A_hat, self.clip_logits_value[0], self.clip_logits_value[1])

        # update attention logits with edges
        H_hat = A_hat  # l,l,h
        if self.edge_input:
            H_hat = H_hat + E  # l,l,h
        # update attention logits with masks
        H_hat_ = H_hat.clone()
        G_ = G.clone()
        # if mask is not None:
        #     mask_ = (mask[:, None, :, None].type(H_hat.dtype) - 1) * 1e9  # l -> 1,l,1
        #     H_hat_ = H_hat_ + mask_
        #     G_ = G_ + mask_

        # if self.attn_mask:
        #     if not M.dtype == H_hat.dtype:
        #         M = M.type(H_hat.dtype)
        #     M_ = (M - 1) * 1e9
        #     H_hat_ = H_hat_ + M_
        #     G_ = G_ + M_

        if self.random_mask_prob > 0.0 and self.training:
            uniform_noise = torch.rand(H_hat_.size(), dtype=H_hat_.dtype, device=H_hat_.device)
            random_mask_ = torch.where(uniform_noise < self.random_mask_prob, -1e9, 0.)
            H_hat_ = H_hat_ + random_mask_
            G_ = G_ + random_mask_

        # form attention weights
        A_tild = F.softmax(H_hat_, dim=1)  # l,l,h
        gates = torch.sigmoid(G_)
        A_tild = A_tild * gates

        # attention output
        if self.attn_dropout > 0.0 and self.training:
            A_tild = F.dropout(A_tild, self.attn_dropout)

        # form output
        V_att = torch.einsum('lmh,mdh->ldh', A_tild, V)
        
         # scale degree
        if self.scale_degree:
            degrees = gates.sum(dim=1)
        if self.scale_degree:
            degrees = gates.sum(dim=1, keepdim=True)  # l,1,h
            if self.scaler_type == 'log':
                degree_scalers = torch.log(1 + degrees)  # l,1,h
            elif self.scaler_type == 'linear':
                degree_scalers = degrees
            else:
                raise ValueError(f'Unknown scaler type {self.scaler_type}')
            V_att = V_att * degree_scalers

        # reshape output
        V_att = V_att.reshape(QKV_shape[0], dot_dim * self.num_heads)  # l,dh
        V_att = V_att.contiguous()


        return V_att, H_hat, A_tild