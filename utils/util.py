import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import h5py
from torch.utils.data import Dataset
import pandas as pd

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class MinMaxScaler():
    """
    Standard the input
    """

    def __init__(self, max, min):
        self.max = max
        self.min = min

    def transform(self, data):
        return (data - self.min) / (self.max-self.min)

    def inverse_transform(self, data):
        return (data * (self.max-self.min)) + self.min

###########################################begin of metanet
def load_h5(filename, keywords):
    f = h5py.File(filename, 'r')
    data = []
    for name in keywords:
        data.append(np.array(f[name]))
    f.close()
    if len(data) == 1:
        return data[0]
    return data


def get_distance_matrix(loc):
    n = loc.shape[0]

    loc_1 = np.tile(np.reshape(loc, (n, 1, 2)), (1, n, 1)) * np.pi / 180.0
    loc_2 = np.tile(np.reshape(loc, (1, n, 2)), (n, 1, 1)) * np.pi / 180.0

    loc_diff = loc_1 - loc_2

    dist = 2.0 * np.arcsin(
        np.sqrt(np.sin(loc_diff[:, :, 0] / 2) ** 2 + np.cos(loc_1[:, :, 0]) * np.cos(loc_2[:, :, 0]) * np.sin(
            loc_diff[:, :, 1] / 2) ** 2)
    )
    dist = dist * 6378.137 * 10000000 / 10000
    return dist


def build_graph(station_map, station_loc, n_neighbors):
    dist = get_distance_matrix(station_loc)

    n = station_map.shape[0]
    src, dst = [], []
    for i in range(n):
        src += list(np.argsort(dist[:, i])[:n_neighbors + 1])
        dst += [i] * (n_neighbors + 1)

    mask = np.zeros((n, n))
    mask[src, dst] = 1
    dist[mask == 0] = np.inf

    values = dist.flatten()
    values = values[values != np.inf]

    dist_mean = np.mean(values)
    dist_std = np.std(values)
    dist = np.exp(-(dist - dist_mean) / dist_std)

    return dist, src, dst


def fill_missing(data):
    T, N, D = data.shape
    data = np.reshape(data, (T, N * D))
    df = pd.DataFrame(data)
    df = df.fillna(method='pad')
    df = df.fillna(method='bfill')
    data = df.values
    data = np.reshape(data, (T, N, D))
    data[np.isnan(data)] = 0
    return data

#########################################end of meta net

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    #d-1/2 * A * d-1/2
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    #(1024,)
    #按行求和
    rowsum = np.array(adj.sum(1)).flatten()
    #D-1 行归一化
    d_inv = np.power(rowsum, -1).flatten()
    #去除 inf
    d_inv[np.isinf(d_inv)] = 0.
    #得到对角矩阵
    d_mat= sp.diags(d_inv)
    #d-1 * A
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def get_graph(filename):
    # (1024, 1024, 32)
    adj_feature = load_h5(filename, ['data'])

    adj_feature = np.sum(adj_feature,axis=-1)
    adj_feature[adj_feature>0] = 1
    adj_feature[adj_feature<=0] = 0

    return adj_feature

def load_adj(filename, adjtype):
    adj_mx = np.load(filename)
    adj_mx = adj_mx['data']

    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        #d-1A,d-1AT
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    # loss = torch.abs(preds - labels)/labels
    #modified
    loss = torch.abs((preds-labels)/labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


def load_h5(filename,keywords):
    f =h5py.File(filename,'r')
    data = []
    for name in keywords:
        data.append(np.array(f[name]))
    f.close()
    if len(data)==1:
        return data[0]
    return data



class LoadData(Dataset):
    def __init__(self,X,Y,timestamp,external=None):
        '''
        :param X: sample (b,n,f,t)
        :param Y: label (b,n,f,t')
        :param external: external info (b,n,f)
        '''

        self.data = torch.from_numpy(X).type(torch.FloatTensor)#(B,T,N,F)
        self.label =  torch.from_numpy(Y).type(torch.FloatTensor) #(B,T,N,F)
        self.timestamp = torch.from_numpy(timestamp).type(torch.FloatTensor) #(B,T,N,F)
        if external is not None:
            self.external =  torch.from_numpy(external).type(torch.FloatTensor) #(B,T,N,F)
            self.is_external = True
        else:
            self.is_external = False


    def __len__(self):
        """
        :return: length of dataset (number of samples).
        """
        return self.data.shape[0]

    def __getitem__(self, index):  # (x, y), index = [0, L1 - 1]
        """
        :param index: int, range between [0, length - 1].
        :return:
            x: torch.tensor, (B,T,N,F)
            y: torch.tensor, (B,T,N,F)
            timestamp: torch.tensor, (B,T,N,F)
            external:torch.tensor,(B,T,N,F)

        """
        x = self.data[index]
        y = self.label[index]
        timestamp = self.timestamp[index]

        if self.is_external:
            external = self.external[index]

            return {"x": x, "y": y,"timestamp":timestamp,"external":external}
        else:
            return {"x": x, "y": y,"timestamp":timestamp}



def load_dataset(filename,num_of_hours,num_of_days,num_of_weeks,batch_size,shuffle=True,
                 external_filename=None,external=False):
    '''
    数据加载函数，会将x,y归一化到[-1,1] x = x-mean/std
    从数据文件加载出来的都是归一化后的数据，以及mean和std,用于还原计算loss
    hour,week,day时间是串起来的
    :param filename: str
    :param num_of_nodes: int
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :param shuffle:
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_target_tensor: (B, N_nodes, 1,T_output)
    '''

    file = os.path.basename(filename) #返回除路径外的文件名
    file = file.split('.')[0] #nyctaxi
    #'./data/nyctaxi'
    dirpath = os.path.dirname(filename)
    loadfile = os.path.join(dirpath,file+'_h'+str(num_of_hours)+'_d'+str(num_of_days)+'_w'+str(num_of_weeks))+'_gwnet'+'.npz'
    print('load file',loadfile)

    #读取数据
    file_data = np.load(loadfile)

    #poi数据
    poi_data = load_h5(external_filename,['data'])

    #获取训练、验证、测试数据
    #todo timeofday,dayofweek等数据

    #to(device)
    train_x = file_data['train_x'] #(b,t,n,c)
    # train_x = train_x[:,:,0:1,:]
    train_target = file_data['train_target'] #(b,t,n,c)
    train_timestamp = file_data['train_timestamp'] #(b,t,n,c)


    val_x = file_data['val_x']
    # train_x = train_x[:, :, 0:1, :]
    val_target = file_data['val_target']
    val_timestamp = file_data['val_timestamp']


    test_x = file_data['test_x']
    # train_x = train_x[:, :, 0:1, :]
    test_target = file_data['test_target']
    test_timestamp = file_data['test_timestamp']

    mean = file_data['mean']
    std = file_data['std']
    mean = torch.from_numpy(mean).type(torch.FloatTensor)
    std = torch.from_numpy(std).type(torch.FloatTensor)

    scaler = StandardScaler(mean,std)

    if external:
        train_dataset = LoadData(train_x, train_target, train_timestamp, np.tile(poi_data,(train_x.shape[0],1,1)))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        # -------val_loader-------
        val_dataset = LoadData(val_x, val_target, val_timestamp, np.tile(poi_data,(val_x.shape[0],1,1)))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
        # -------test_loader-------
        test_dataset = LoadData(test_x, test_target, test_timestamp, np.tile(poi_data,(test_x.shape[0],1,1)))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        train_dataset = LoadData(train_x, train_target, train_timestamp, None)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        # -------val_loader-------
        val_dataset = LoadData(val_x, val_target, val_timestamp, None)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
        # -------test_loader-------
        test_dataset = LoadData(test_x, test_target, test_timestamp,None)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader,val_loader,test_loader,scaler
