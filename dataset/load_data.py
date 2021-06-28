import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import util,generate_time_embedding



class MyDataset(Dataset):
    def __init__(self,X,Y,timestamp,external):
        '''
        :param X: sample (b,f,n,t)
        :param Y: label (b,f,n,t)
        :param timestamp: timestamp (b,f,n,t)
        :param external: external info (b,f,n,t)
        '''
        self.data = torch.from_numpy(X).type(torch.FloatTensor)
        self.label =  torch.from_numpy(Y).type(torch.FloatTensor)
        self.timestamp = torch.from_numpy(timestamp).type(torch.FloatTensor)
        self.external =  torch.from_numpy(external).type(torch.FloatTensor)


    def __len__(self):
        """
        :return: length of dataset (number of samples).
        """
        return self.data.shape[0]

    def __getitem__(self, index):  # (x, y), index = [0, L1 - 1]
        """
        :param index: int, range between [0, length - 1].
        :return:
            x: torch.tensor, (B,F,N,T)
            y: torch.tensor, (B,F,N,T)
            timestamp: torch.tensor, (B,F,N,T)
            external:torch.tensor,(B,F,N,T)
        """
        x = self.data[index]
        y = self.label[index]
        timestamp = self.timestamp[index]
        external = self.external[index]

        return {"x": x, "y": y,"timestamp":timestamp,"external":external}

def dataloader(data_path,train_prop,eval_prop):
    data = util.load_h5(os.path.join(data_path, 'BJ_FLOW.h5'), ['data'])
    # time_data = np.load(os.path.join(data_path, 'BJ_TIME.npy'))

    time_data = generate_time_embedding.generate_time()

    days, hours, rows, cols, _ = data.shape
    num_nodes = rows * cols
    data = np.reshape(data, (days * hours, num_nodes, -1))
    time_data = np.expand_dims(time_data,1).repeat(num_nodes,1)

    n_timestamp = data.shape[0]

    num_train = int(n_timestamp * train_prop)
    num_eval = int(n_timestamp * eval_prop)
    num_test = n_timestamp - num_train - num_eval

    return data[:num_train], data[num_train: num_train + num_eval], data[-num_test:],\
           time_data[:num_train],time_data[num_train: num_train + num_eval], time_data[-num_test:]

def get_geo_feature(datapath):
    geo = util.load_h5(os.path.join(datapath, 'BJ_FEATURE.h5'), ['embeddings'])
    row, col, _ = geo.shape
    geo = np.reshape(geo, (row * col, -1))

    geo = (geo - np.mean(geo, axis=0)) / (np.std(geo, axis=0) + 1e-8)
    return geo


def dataiter_all_sensors_seq2seq(flow,time_data,scaler, input_len,output_len,batch_size,geo_path,
                                 shuffle=True):

    mask = np.sum(flow, axis=(1,2)) > 5000
    flow = scaler.transform(flow)
    n_timestamp, num_nodes, _ = flow.shape

    # geo_feature = get_geo_feature(geo_path) #(1024,940)
    geo_feature = util.load_h5(os.path.join(geo_path,'bj_tfidf_poi.h5'),['data'])
    data,time_list,feature, label  = [], [], [],[]
    for i in range(n_timestamp - input_len - output_len + 1):
        if mask[i + input_len: i + input_len + output_len].sum() != output_len:
            continue

        data.append(flow[i: i + input_len])
        label.append(flow[i + input_len: i + input_len + output_len])
        time_list.append(time_data[i: i + input_len])
        feature.append(geo_feature)

        if i % 1000 == 0:
            logging.info('Processing %d timestamps', i)
        # if i > 0: break

    data = np.stack(data).swapaxes (1,3)                       # [B, D, N, T]
    label = np.stack(label).swapaxes (1,3)                     # [B, D, N, T]
    time_arr = np.stack(time_list).swapaxes (1,3)              # [B, D, N, T]
    feature = np.stack(feature)                                # [B, N, D]

    logging.info('shape of feature: %s', feature.shape)
    logging.info('shape of data: %s', data.shape)
    logging.info('shape of time: %s', time_arr.shape)
    logging.info('shape of label: %s', label.shape)

    dataset = MyDataset(data, label, time_arr, feature)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader

def dataloader_all_seq2seq(data_path,train_prop,eval_prop,input_len,output_len,batch_size):
    train_data, eval_data, test_data,train_time,val_time,test_time\
        = dataloader(data_path,train_prop,eval_prop)

    scaler = util.StandardScaler(np.mean(train_data),np.std(train_data))

    train_loader = dataiter_all_sensors_seq2seq(train_data,train_time,scaler,input_len,output_len,batch_size,data_path)
    val_loader = dataiter_all_sensors_seq2seq(eval_data,val_time, scaler, input_len,output_len,batch_size,data_path, shuffle=False)
    test_loader = dataiter_all_sensors_seq2seq(test_data,test_time, scaler, input_len,output_len,batch_size,data_path, shuffle=False)

    return train_loader,val_loader,test_loader,scaler


