import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
import torch.nn as nn
import math
    
class _prob_gate(nn.Module):
    def __init__(self, in_size=512, hidden_size=16):  # hidden size used to be 128
        super(_prob_gate, self).__init__()
        self.g = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 2)
        )
    def gumbel_softmax(self, logits, temperature=10.0, eps=1e-10):
        y = F.softmax(logits, dim=-1)
        return y
    
    def forward(self, x):
        x = self.g(x)
        x = self.gumbel_softmax(x)
        return x

class prob_gate(nn.Module):
    def __init__(self, in_size=512, out_channel=9):
        super(prob_gate, self).__init__()
        self.out_channel = out_channel
        self.in_size = in_size
        self.gates = nn.ModuleList([_prob_gate(in_size) for _ in range(out_channel-1)])
        # for scalable use
        self.extra_gates = nn.ModuleList([])
    
    def forward(self, x, t_gt_type):
        # x         : input feature
        # t_gt_type : gt degradation type, only used for visualization
        if isinstance(t_gt_type, int):
            gt_type = t_gt_type
        else:
            gt_types= torch.split(t_gt_type, split_size_or_sections=1, dim=-1)
            gt_type = gt_types[0].item()
        p_list = []
        value_list = []
        _wandb_dict = {}
        for gate in self.gates:
            p_list.append(gate(x))
        for gate in self.extra_gates:
            p_list.append(gate(x))
        
        accept, refuse = torch.split(p_list[0], split_size_or_sections=1, dim=-1)
        value_list.append(accept)

        refuse_all = torch.zeros_like(refuse)
        refuse_all += refuse

        for i in range(1, len(p_list)):
            accept, refuse = torch.split(p_list[i], split_size_or_sections=1, dim=-1)
            refuse_all += refuse
            value_list.append(accept)

        # control the indolent gate value between 0-1
        refuse_all /= len(self.gates) + len(self.extra_gates)

        value_list.insert(0, refuse_all)
        g_values = torch.cat(value_list, dim=-1)
        g_values = F.softmax(g_values, dim=-1)
        g_values = g_values.squeeze(1)

        for i, accept in enumerate(value_list):
            _wandb_dict['gate_'+str(i)] = torch.mean(accept).item()
            _wandb_dict['gate_'+str(gt_type)+'_'+str(i)]= torch.mean(accept).item()

        wandb.log(_wandb_dict)
        return g_values
    
class expert(nn.Module):
    def __init__(self, in_size, indolent=False, hidden_size = 16):
        super(expert, self).__init__()
        self.indolent = indolent
        if not indolent:
            self.model = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, in_size)
                )
            self.init_weights()

    def init_weights(self):
        flag = False
        for m in self.model:
            if type(m) == nn.Linear:
                if flag:
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    nn.init.zeros_(m.bias)
                

    def forward(self, x):
        if self.indolent:
            return x
        return self.model(x)

class expert_aff(nn.Module):
    def __init__(self, in_size, indolent=False):
        super(expert_aff, self).__init__()
        self.indolent = indolent
        if not indolent:
            self.weight = nn.Parameter(torch.randn(in_size))
            self.bias = nn.Parameter(torch.randn(in_size))
            self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)
                
    def forward(self, x):
        if self.indolent:
            return x
        else:
            x = x * self.weight + self.bias
        return x



class RouGE(nn.Module):
    def __init__(self, in_size, num_expert):
        super(RouGE, self).__init__()
        # print('RouGE')
        self.gate = prob_gate(out_channel = num_expert)
        expert_list = []
        indo_expert = expert(in_size=in_size, indolent=True)  # adapter-like
        # indo_expert = expert_aff(in_size=in_size, indolent=True)   # affine-expert
        expert_list.append(indo_expert)
        for _ in range(num_expert-1):
            exp = expert(in_size=in_size, indolent=False)   # adapter-like
            # exp = expert_aff(in_size=in_size, indolent=False)   # affine-expert
            expert_list.append(exp)
        self.experts = nn.ModuleList(expert_list)

    def forward(self, x, feature, gt_type):
        # x         : input feature
        # feature   : image feature
        # gt_type : gt degradation type, only used for visualization

        expert_values = []
        g_values = self.gate(feature, gt_type)
        for i in range(len(self.experts)):
            expert_values.append(self.experts[i](x).unsqueeze(1))
        values = torch.cat(expert_values, dim=1).permute(2,3,0,1)
        values = torch.mul(g_values, values).permute(2,3,0,1)
        values = torch.sum(values, dim=1)
        
        return values


# control the adaptor used in sam image encoder
def build_RouGE(in_size, num_expert = 6):
    # in_size: input feature size
    return RouGE(in_size, num_expert = num_expert)

