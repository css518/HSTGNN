import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layer import TemporalConv,GraphConv

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)

class GLU(nn.Module):
    def __init__(self,in_dim,dropout_rate=None):
        super(GLU,self).__init__()
        self.mlp1 = linear(in_dim,in_dim)
        self.mlp2 = linear(in_dim,in_dim)
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        '''
        :param x: (b,c,n,t)
        :return:
        '''
        if self.dropout_rate is not None:
            x = self.dropout(x)
        left = self.mlp1(x)
        right = self.mlp2(x)
        out = self.sigmoid(left) * right
        return out

class MLP(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,bias=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim,hidden_dim,bias=bias)
        self.fc2 = nn.Linear(hidden_dim,out_dim,bias=bias)
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        return out

class GatedFusion(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,num_nodes,dropout_rate):
        super(GatedFusion, self).__init__()
        self.fc1 = linear(in_dim,hidden_dim)
        self.fc2= linear(hidden_dim,out_dim)

        self.gate = GLU(out_dim,dropout_rate)
        self.elu = nn.ELU()
        self.layer_norm = nn.LayerNorm([num_nodes,out_dim])

    def forward(self,x,auxiliary):
        '''
        :param x: primary inuput,(b,c,n,t)
        :param auxiliary:auxiliary input,(b,c,n,t)
        :return:
        '''
        residual = x
        x = torch.cat([x,auxiliary],dim=1)
        hidden = self.fc1(x)
        #n2 = elu(hidden)
        hidden = self.elu(hidden)
        #n1 = ELU(w1n1 + b1)
        hidden = self.fc2(hidden)
        #GLU
        gate = self.gate(hidden)
        #addnorm
        out = residual + gate
        out = self.layer_norm(out.transpose(1,3))
        return out.transpose(1,3)

class CrossST(nn.Module):
    def __init__(self,in_dim,out_dim,external_dim,num_nodes,dropout_rate,gated=True):
        super(CrossST, self).__init__()

        if gated:
            self.spatial_fusion = GatedFusion(2*in_dim,2*in_dim,out_dim,num_nodes,dropout_rate)
            self.temporal_fusion = GatedFusion(2*in_dim,2*in_dim,out_dim,num_nodes,dropout_rate)
            self.all_fusion = GatedFusion(3*in_dim,2*in_dim,out_dim,num_nodes,dropout_rate)
        else:
            self.fc1 = linear(2*in_dim,out_dim)
            self.fc2 = linear(2*in_dim,out_dim)
            self.fc3 = linear(3*in_dim,out_dim)

        self.gated =gated
        self.mlp = MLP(external_dim,2*in_dim,out_dim)

    def forward(self,x,time_embedding,external):
        '''
        :param x: (b,c,n,t)
        :param time_embedding: (b,c,n,t)
        :param external: (b,c,n)
        :return:
        '''
        #(b,n,c)
        external = self.mlp(external)
        external = external.unsqueeze(1).repeat((1,x.shape[-1],1,1)) #(b,t,n,c)
        external = external.transpose(1,3)

        if self.gated:
            spatial = self.spatial_fusion(x,external)
            temporal = self.temporal_fusion(x,time_embedding)
            all = self.all_fusion(x,torch.cat([time_embedding,external],dim=1))
        else:
            spatial = self.fc1(torch.cat([x,external],dim=1))
            temporal = self.fc2(torch.cat([x,time_embedding],dim=1))
            all = self.fc3(torch.cat([x,time_embedding,external],dim=1))

        return spatial,temporal,all

class STConvLayer(nn.Module):
    def __init__(self,in_dim,out_dim,kernel_size,num_nodes,n_supports):
        super(STConvLayer, self).__init__()
        self.tconv = TemporalConv(in_dim,out_dim,kernel_size,stride=1,dilation=2,num_nodes=num_nodes)
        self.gconv = GraphConv(in_dim,out_dim,num_nodes,n_supports,max_step=2)

    def forward(self,spatial,temporal,supports):
        b,c,n,t = spatial.shape

        #spatial
        s_out = []
        for k in range(t):
            graph_signal = spatial[...,k] #(b,n,c)
            graph_signal = self.gconv(graph_signal,supports)
            graph_signal = graph_signal.unsqueeze(-1) #(b,n,c,1)
            s_out.append(graph_signal)

        s_out = torch.cat(s_out,dim=-1)
        s_out = s_out.transpose(1,2)

        #temporal
        t_out = self.tconv(temporal)
        s_out = s_out[...,-t_out.shape[-1]:]

        return s_out,t_out


class HSTGNN(nn.Module):
    def __init__(self,support,in_dim,hidden_dim,layers,num_nodes,external_dim,input_len,
                 output_len,dropout_rate,gated,multi_task):
        super(HSTGNN, self).__init__()

        self.start_conv = linear(in_dim,hidden_dim)
        self.mix_layer = CrossST(hidden_dim,hidden_dim,external_dim,num_nodes,dropout_rate,gated=gated)

        self.st_layers = nn.ModuleList()
        for i in range(layers):
            self.st_layers.append(STConvLayer(hidden_dim,hidden_dim,kernel_size=3,num_nodes=num_nodes,
                                          n_supports=3))
        self.layers = layers
        Ko = input_len - self.layers * 2
        if Ko >= 1:
            pass
        else:
            raise ValueError('ERROR: kernel size Ko must be greater than, but received.')

        self.mlp = linear(input_len,Ko)
        self.mlp_t = linear(Ko,output_len)
        self.mlp_s = linear(hidden_dim,2)

        m, p, n = torch.svd(support)
        initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
        initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
        self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
        self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)

        self.w1 = nn.Parameter(torch.eye(10), requires_grad=True)
        self.w2 = nn.Parameter(torch.eye(10), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(10), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(10), requires_grad=True)

        self.adp1 = None
        self.adp2 = None
        self.adp3 = None
        self.multi_task = multi_task

    def forward(self,x,timestamps,external):
        '''
        x: traffic flow ∈ (b,c,n,t)
        timestamps: time embedding ∈(b,c,n,t)
        external: POI ∈(b,c,n)
        '''
        nodevec1 = self.nodevec1        #N×1
        nodevec2 = self.nodevec2        #1×N
        n = nodevec1.size(0)
        supports = []
        self.adp1 = F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=1)  #N×N
        supports.append(self.adp1)

        nodevec1 = nodevec1.mm(self.w1) +self.b1.repeat(n,1)
        nodevec2 = (nodevec2.T.mm(self.w1) + self.b1.repeat(n, 1)).T
        self.adp2 = F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=1)
        supports.append(self.adp2)

        nodevec1 = nodevec1.mm(self.w2) + self.b2.repeat(n, 1)
        nodevec2 = (nodevec2.T.mm(self.w2) + self.b2.repeat(n, 1)).T
        self.adp3 = F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=1)
        supports.append(self.adp3)

        #train
        x = self.start_conv(x)

        spatial,temporal,all_mixed = self.mix_layer(x,timestamps,external)

        for i in range(self.layers):
            spatial,temporal = self.st_layers[i](spatial,temporal,supports)

        all_mixed = self.mlp(all_mixed.transpose(1,3))
        out = torch.add(torch.add(spatial,temporal),all_mixed.transpose(1,3))

        if not self.multi_task:
            out = self.mlp_s(out)
        out = self.mlp_t(out.transpose(1,3))

        return out.transpose(1,3)

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num, model,scaler,c_in,loss_func):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        self.task_num = task_num
        self.scaler = scaler

        self.w1 = 0.25
        self.w2 = 0.25
        self.w3 = 0.5

        self.conv_list = nn.ModuleList()
        self.conv_list.append(nn.Conv2d(c_in,1,kernel_size=1))
        self.conv_list.append(nn.Conv2d(c_in,1,kernel_size=1))
        self.conv_list.append(nn.Conv2d(c_in,2,kernel_size=1))
        self.loss_func = loss_func


    def forward(self, input, targets,timestamp,external):
        '''
        :param input: (b,c,n,t)
        :param targets: label (b,2,n,3)
        :return: loss,
        predict,
        real_val,
        self.log_vars.data.tolist()
        '''

        #shared high level st feature
        outputs = self.model(input,timestamp,external)

        inflow = self.conv_list[0](outputs).squeeze(1)
        outflow= self.conv_list[1](outputs).squeeze(1)
        allflow = self.conv_list[2](outputs)

        #inverse transform
        predict = self.scaler.inverse_transform(allflow)
        inflow = self.scaler.inverse_transform(inflow)
        outflow = self.scaler.inverse_transform(outflow)

        # (b,c,n,t)
        real_val = self.scaler.inverse_transform(targets)

        loss_inflow =  self.loss_func(inflow,real_val[:,0,:,:])
        loss_outflow =  self.loss_func(outflow,real_val[:,1,:,:])
        loss_allflow =  self.loss_func(predict,real_val)

        loss_all = self.w1*loss_inflow + self.w2*loss_outflow + \
                   self.w3*loss_allflow

        return loss_all,predict,real_val