import torch
import torch.nn as nn

class TemporalConv(nn.Module):
    def __init__(self,in_dim,out_dim,kernel_size,stride,dilation,num_nodes):
        super(TemporalConv, self).__init__()
        self.conv_left = nn.Conv2d(in_dim,out_dim,kernel_size=(1,kernel_size),stride=(1,stride),padding=(0,1),dilation=(1,dilation))
        self.conv_right = nn.Conv2d(in_dim,out_dim,kernel_size=(1,kernel_size),stride=(1,stride),padding=(0,1),dilation=(1,dilation))
        self.layernorm = nn.LayerNorm([num_nodes,out_dim])

    def forward(self,x):
        '''
        :param x:(b,c,n,t)
        :return: (b,c,n,t')
        '''
        left = self.layernorm(self.conv_left(x).transpose(1,3))
        right = self.layernorm(self.conv_right(x).transpose(1,3))
        out = torch.sigmoid(left)*torch.tanh(right)
        return out.transpose(1,3)

class GraphConv(nn.Module):
    def __init__(self,in_dim,out_dim,num_nodes,n_supports,max_step):
        super(GraphConv, self).__init__()
        self._num_nodes = num_nodes
        self._max_diffusion_step =max_step
        num_metrics = max_step * n_supports + 1
        self.out = nn.Linear(in_dim * num_metrics, out_dim)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self,inputs,supports):
        '''
        :param inputs,(b,c,n)
        :param supports:list,(n,n)
        :return:(b,n,c')
        '''
        b,c,n = inputs.shape
        x = inputs
        x0 =x.permute([2,1,0]).reshape(n,-1) #(b,n,c)->(n,c,b)->(n,b*c)
        x = x0.unsqueeze(dim=0) #(1,n,b*c)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                x1 = support.mm(x0)
                x = self._concat(x,x1) #(n_supports,n,b*c)
                for k in range(2,self._max_diffusion_step+1):
                    x2 = 2*support.mm(x1) - x0
                    x = self._concat(x,x2)
                    x1,x0 = x2,x1
        x = x.reshape(-1,n,c,b).transpose(0,3) #(b,n,c,num_matrices)
        x = x.reshape(b,n,-1) #(b,n,c*num_matrices)
        x = self.out(x)
        return x