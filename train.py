import torch
import numpy as np
import argparse
import time
from utils import util
from dataset import load_data
from engine import trainer
import os

parser = argparse.ArgumentParser()
#dataset
parser.add_argument('--datapath',type=str,default='./data',help='data path')
parser.add_argument('--adjdata',type=str,default='./data/cossimi_graph.npz',help='initial adj adjacent matrix')
parser.add_argument('--adjtype',type=str,default='symnadj',help='adj type')
parser.add_argument('--aptonly',action='store_true',default=False,help='whether only adaptive adj')
parser.add_argument('--external_dim',type=int,default=940,help='external information dimension')
parser.add_argument('--input_len',type=int,default=12,help='')
parser.add_argument('--out_len',type=int,default=3,help='')

#model
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--layers',type=int,default=4,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=1024,help='number of nodes')
parser.add_argument('--gated',type=bool,default=True,help='whether to use gated fusion')
parser.add_argument('--multi_task',type=bool,default=True,help='whether to use multitask learning')
parser.add_argument('--is_external',type=bool,default=True,help='')
parser.add_argument('--is_timestamp',type=bool,default=True,help='')

#trainning
parser.add_argument('--epochs',type=int,default=200,help='')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--train_prop',type=int,default=0.8,help='')
parser.add_argument('--eval_prop',type=int,default=0.1,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./garage/metr',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--multi_gpu',type=bool,default=True,help='whether to parallel training')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
torch.backends.cudnn.enabled = False

def main():
    #set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    #load initial adj
    adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = torch.Tensor(adj_mx[0])
    if args.aptonly:
        supports = None

    #data loader
    train_loader,val_loader,test_loader,scaler = load_data.dataloader_all_seq2seq(args.datapath, args.train_prop, args.eval_prop, args.input_len, args.out_len, args.batch_size)

    engine = trainer(supports,scaler,args.input_len, args.out_len, args.num_nodes,args.dropout, args.learning_rate, args.weight_decay,
                     args.layers,args.in_dim,args.nhid,args.external_dim,args.gated,args.multi_task,args.multi_gpu)

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []

    for i in range(1,args.epochs+1):
        # if i % 10 == 0:
        #    lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
        #    for g in engine.optimizer.param_groups:
        #        g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        #шонч╗Г
        for iter, data in enumerate(train_loader):
            x = data['x'].cuda()
            y = data['y'].cuda()

            timestamp,external = None,None
            if args.is_timestamp:
                timestamp = data['timestamp'].cuda() #(b,t,n,c)
            if args.is_external:
                poi_data = data['external'].cuda()

            metrics = engine.train(x, y,timestamp,poi_data)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])

            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)

        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, data in enumerate(val_loader):
            x = data['x'].cuda()
            y = data['y'].cuda()

            timestamp,external = None,None
            if args.is_timestamp:
                timestamp = data['timestamp'].cuda() #(b,t,n,c)

            if args.is_external:
                poi_data = data['external'].cuda()

            metrics = engine.eval(x, y,timestamp,poi_data)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])

        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))

    #test
    outputs = []
    realy = []
    for iter, data in enumerate(test_loader):
        x = data['x'].cuda()
        y = data['y'].cuda()

        timestamp, external = None, None
        if args.is_timestamp:
            timestamp = data['timestamp'].cuda()
        if args.is_external:
            poi_data = data['external'].cuda()

        with torch.no_grad():
            if args.multi_task:
                _,y_hat,y = engine.model(x,y,timestamp,poi_data)
            else:
                y_hat = engine.model(x,timestamp,poi_data)
        outputs.append(y_hat)
        realy.append(y)

    realy = torch.cat(realy,dim=0)
    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    amae = []
    amape = []
    armse = []

    for i in range(args.out_len):
        if args.multi_task: #in multitask wrapper, we have inversed transform the data
            pred = yhat[:,:,:,i]
            real = realy[:,:,:,i]
        else:
            pred = scaler.inverse_transform(yhat[:,:,:,i])
            real = scaler.inverse_transform(realy[:,:,:,i])
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over '+str(args.out_len)+' horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))