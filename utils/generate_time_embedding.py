import pandas as pd
import numpy as np
import time

def timestamp2vec(timestamps):
    # tm_wday range [0, 6], Monday is 0 and sunday is 6
    #7872
    vec = [time.strptime(str(t,encoding='utf-8'), '%Y%m%d%H') for t in timestamps]  # python3

    # vec = [time.strptime(t[:8], '%Y%m%d').tm_wday for t in timestamps]  # python2
    ret = []

    for i in vec:
        #timeofday
        arr = np.zeros(24).tolist()
        hour = i.tm_hour
        arr[hour] = 1
        #day of week
        ii = i.tm_wday
        v = [0 for _ in range(7)]
        v[ii] = 1
        #week or weekend
        if ii >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        arr.extend(v)

        ret.append(arr)
    timestamps = np.asarray(ret)
    # timestamps = np.expand_dims(timestamps,axis=1)
    return timestamps

def complete_time(dt):
    if(dt<10):
        dt = '0'+str(dt)
    else:
        dt = str(dt)
    return dt

def extrac_date_hour(date_range):
    result = []
    for date in date_range:
        year = str(date.year)
        month = complete_time(date.month)
        day = complete_time(date.day)
        hour = complete_time(date.hour)
        temp_str = year+month+day+hour
        temp_str = str.encode(temp_str)
        result.append(temp_str)
    return result

def generate_time():
    date_range = pd.date_range(start='2015-02-01 00:00:00',end='2015-07-01 23:00:00',freq='h',normalize=True,closed='left')
    #提取日期
    result = extrac_date_hour(date_range)
    #获取one-hot编码
    timestamps = timestamp2vec(result)
    return timestamps

def get_similarity_adj(data,norm=True):
    '''
    :param data: (num_nodes,feature_num)
    :return: adj(num_nodes,num_nodes)
    '''
    num_nodes,num_feature = data.shape
    arr = np.zeros((num_nodes,num_nodes))

    for i in range(num_nodes):
        x = data[i]
        arr[i][i] = 1
        for j in range(i+1,num_nodes):
            y = data[j]
            dot_product, square_sum_x, square_sum_y = 0, 0, 0

            for k in range(num_feature):
                dot_product += x[k]*y[k]
                square_sum_x += x[k]*x[k]
                square_sum_y += y[k]*y[k]
            cos = dot_product/(np.sqrt(square_sum_x)*np.sqrt(square_sum_y))
            if(cos >= 0.6):
                arr[i][j] = arr[j][i] = 1
            else:
                arr[i][j] = arr[j][i] = 0

    print(arr.shape)
    return arr

if __name__ == '__main__':
    #输入日期范围
    date_range = pd.date_range(start='2015-02-01 00:00:00',end='2015-07-01 23:00:00',freq='h',normalize=True,closed='left')
    filename = '../data/BJ_TIME.npy'
    num_nodes = 1024
    #提取日期
    result = extrac_date_hour(date_range)
    #获取one-hot编码
    timestamps = timestamp2vec(result)

    print(timestamps.shape) #（timesteps,32）,32=24+7+1,24小时，7天，1是否周末

    # np.save(filename,data=timestamps)