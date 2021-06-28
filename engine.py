import torch.optim as optim
from model.model import *
from utils import util

class trainer():
    def __init__(self,support, scaler,input_len,output_len,num_nodes,dropout,lrate,wdecay
                 ,layers,in_dim,hidden_dim,external_dim,gated,multi_task,multi_gpu):

        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5
        self.multi_task = multi_task
        self.multi_gpu = multi_gpu

        model = HSTGNN(support,in_dim,hidden_dim,layers,num_nodes,external_dim,input_len,output_len,dropout,gated,multi_task)

        if multi_task:
            self.model = MultiTaskLossWrapper(task_num=3, model= model,scaler=self.scaler,c_in=hidden_dim,
                                              loss_func= self.loss)
        else:
            self.model = model

        if multi_gpu:
            self.model = nn.DataParallel(self.model)

        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)


    def train(self, input, real_val,timestamp=None,external=None):
        self.model.train()

        if self.multi_task:
            loss, predict, real_val = self.model(input,real_val,timestamp,external)
            #for multitask learning with multi-gpu,it will return two loss and concat to a two dimension tensor
            if self.multi_gpu:
                loss = loss.sum()
        else:
            predict =  self.model(input,timestamp,external)
            predict = self.scaler.inverse_transform(predict)
            real_val = self.scaler.inverse_transform(real_val)
            loss = self.loss(predict,real_val)

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict, real_val, 0.0).item()
        rmse = util.masked_rmse(predict, real_val, 0.0).item()

        return loss.item(),mape,rmse

    def eval(self,input,real_val,timestamp,external):
        self.model.eval()

        if self.multi_task:
            loss, predict, real_val  = self.model(input,real_val,timestamp,external)
            #for multitask learning with multi-gpu,it will return two loss and concat to a two dimension tensor
            if self.multi_gpu:
                loss = loss.sum()
        else:
            predict =  self.model(input,timestamp,external)
            predict = self.scaler.inverse_transform(predict)
            real_val = self.scaler.inverse_transform(real_val)
            loss = self.loss(predict,real_val)

        mape = util.masked_mape(predict, real_val, 0.0).item()
        rmse = util.masked_rmse(predict, real_val, 0.0).item()
        return loss.item(),mape,rmse
