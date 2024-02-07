import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from einops import rearrange

import math
from torch_geometric.typing import OptTensor
from tsl.nn.models.base_model import BaseModel




class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    @staticmethod
    def compute_delta(mask, freq=1):
        delta = torch.zeros_like(mask).float()
        one_step = torch.tensor(freq, dtype=delta.dtype, device=delta.device)
        for i in range(1, delta.shape[-2]):
            m = mask[..., i - 1, :]
            delta[..., i, :] = m * one_step + (1 - m) * torch.add(delta[..., i - 1, :], freq)
        return delta
    
    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class RITSModel(nn.Module):
    def __init__(self, 
                 input_size,
                 rnn_hid_size):
        super(RITSModel, self).__init__()

        self.input_size = input_size
        self.rnn_hid_size = rnn_hid_size

        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(self.input_size * 2, 
                                    self.rnn_hid_size)
        
        self.temp_decay_h = TemporalDecay(input_size = self.input_size, 
                                          output_size = self.rnn_hid_size, 
                                          diag = False)
        self.temp_decay_x = TemporalDecay(input_size = self.input_size, 
                                          output_size = self.input_size, 
                                          diag = True)
        
        self.hist_reg = nn.Linear(self.rnn_hid_size, 
                                  self.input_size)
        self.feat_reg = FeatureRegression(self.input_size)
        
        self.weight_combine = nn.Linear(self.input_size * 2, 
                                        self.input_size)

    def init_hidden_states(self, x):
        return Variable(torch.zeros((x.shape[0], self.rnn_hid_size))).to(x.device)
    
    def forward(self, x, mask):
        # Original sequence with 24 time steps
        values = x
        masks = torch.ones_like(x, dtype=torch.uint8) if mask is None else mask
        deltas = TemporalDecay.compute_delta(masks)

        # init rnn states
        h = self.init_hidden_states(x)
        c = self.init_hidden_states(x)

        x_out = []
        first, second, third = [], [], []
        steps = x.size()[1]
        for t in range(steps):
            # x: [batch, time, features]
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)
            first.append(x_h.unsqueeze(dim = 1)) # for loss calculation with x_h
            
            x_c = torch.where(m.bool(), x, x_h)       

            z_h = self.feat_reg(x_c)
            second.append(z_h.unsqueeze(dim = 1)) # for loss calculation with z_h

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            third.append(c_h.unsqueeze(dim = 1)) # for loss calculation with c_h

            c_c = torch.where(m.bool(), x, c_h)   

            inputs = torch.cat([c_c, m], dim = 1)

            # h = self.rnn_cell(inputs, h) # if  GRU
            h, c = self.rnn_cell(inputs, (h, c)) # if LSTM

            x_out.append(c_h.unsqueeze(dim = 1)) # last c_h is the imputation

        out = {'first': torch.cat(first, dim = 1),
               'second': torch.cat(second, dim = 1),
               'third': torch.cat(third, dim = 1),
               'x_out': torch.cat(x_out, dim = 1)}

        return out




class BRITSModel(BaseModel):
    def __init__(self, 
                 input_size,
                 rnn_hidden_size):
        super(BRITSModel, self).__init__()

        self.rits_f = RITSModel(input_size, rnn_hidden_size)
        self.rits_b = RITSModel(input_size, rnn_hidden_size)

    def forward(self, 
                x: Tensor,
                mask: OptTensor = None):
        _, _, nodes, _ = x.size()

        # # # Whiten missing values
        # x = x * mask 
        # # handle nan if still present
        # if torch.isnan(x).any():
        #     x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        # reshape
        x = rearrange(x, 'b t n f -> (b n) t f')
        mask = rearrange(mask, 'b t n f -> (b n) t f').to(torch.int)

        ret_f = self.rits_f(x, mask)
        x_rev = self.reverse({'x': x})['x']
        mask_rev = self.reverse({'mask': mask})['mask']
        ret_b = self.reverse(self.rits_b(x_rev, mask_rev))
        ret = self.merge_ret(ret_f, ret_b)

        # reshape
        ret['imputed_data'] = rearrange(ret['imputed_data'], '(b n) t f -> b t n f', n=nodes) 
        # rest stay shaped, ground truth is reshaped to (b n) t f in the loss calculation
        return ret

    def merge_ret(self, ret_f, ret_b):
        imputations = (ret_f['x_out'] + ret_b['x_out']) / 2

        ret = {"model_name": "BRITS",
               "imputed_data": imputations,
                "imputed_data_fwd": ret_f['x_out'],     # for consistency loss
                "imputed_data_bwd": ret_b['x_out'],    # for consistency loss
                "first_fwd": ret_f['first'],       # for loss calculation with x_h
                "second_fwd": ret_f['second'],     # for loss calculation with z_h
                "third_fwd": ret_f['third'],       # for loss calculation with c_h
                "first_bwd": ret_b['first'],       # for loss calculation with x_h
                "second_bwd": ret_b['second'],     # for loss calculation with z_h
                "third_bwd": ret_b['third']        # for loss calculation with c_h
        }

        return ret

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad = False)

            if torch.cuda.is_available():
                indices = indices.cuda()

            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

