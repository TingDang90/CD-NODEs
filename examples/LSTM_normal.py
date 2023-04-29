import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.io as io
import scipy.signal as sg

from torch.autograd import Variable
#import torch.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--inp_dim', type=int, default=100)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--train_dir', type=str, default=None)
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

def chunking_frame(data,window,overlap):
    shift = window - overlap
    frame_num = int(np.ceil(data.shape[0] / float(shift)))
    chunked_data = np.zeros((frame_num, window, data.shape[1]), dtype='float')
    #rnum=np.flip(data[-window:-1, :],axis=0)
    #data=np.append(data,rnum, axis=0)
    i_w = 0
    for wn in range(frame_num):
        if (wn == frame_num - 1):
            chunked_data[wn, 0:np.shape(data[i_w:, :])[0], :] = data[i_w:, :]
        else:
            chunked_data[wn, 0:window, :] = data[i_w:i_w+window,:]
        i_w = i_w+shift

    return chunked_data


def Binomialfilter(delay_frame):
    h = [0.5, 0.5]
    j = int(2 * delay_frame + 1)
    binomialcoff = np.convolve(h, h)
    for i in range(j - 3):
        binomialcoff = np.convolve(binomialcoff, h)
    binomialcoff = binomialcoff / (np.sum(binomialcoff))
    return binomialcoff

def Rmseloss(pred_y, true_y):
    return ((pred_y - true_y) ** 2)

def CCCloss(pred_y, true_y):
    ux = torch.mean(pred_y)
    uy = torch.mean(true_y)
    pred_y1 = pred_y - ux
    true_y1 = true_y - uy
    cc = torch.sum(pred_y1 * true_y1) / (
                torch.sqrt(torch.sum(pred_y1 ** 2)) * torch.sqrt(torch.sum(true_y1 ** 2)))
    if device == 'gpu':
        cc_num = torch.cuda.FloatTensor(1, ).fill_(2.0) * cc * (
                torch.sqrt(torch.mean(pred_y1 ** 2)) * torch.sqrt(torch.mean(true_y1 ** 2))).to(device)
        ccc = cc_num / (torch.mean(pred_y1 ** 2) + torch.mean(true_y1 ** 2) + torch.sum((ux - uy) ** 2))
        ccc = torch.cuda.FloatTensor(1, ).fill_(1.0) - ccc
    else:
        cc_num = torch.tensor(2.0) * cc * (
                torch.sqrt(torch.mean(pred_y1 ** 2)) * torch.sqrt(torch.mean(true_y1 ** 2))).to(device)
        ccc = cc_num / (torch.mean(pred_y1 ** 2) + torch.mean(true_y1 ** 2) + torch.sum((ux - uy) ** 2))
        ccc = torch.tensor(1.0) - ccc

    return ccc


def cc_cal(pred_y, true_y):
    ux = torch.mean(pred_y)
    uy = torch.mean(true_y)
    pred_y1 = pred_y - ux
    true_y1 = true_y - uy
    cc = torch.sum(pred_y1 * true_y1) / (
            torch.sqrt(torch.sum(pred_y1 ** 2)) * torch.sqrt(torch.sum(true_y1 ** 2)))

    cc_num = 2.0 * cc * (torch.sqrt(torch.mean(pred_y1 ** 2)) * torch.sqrt(torch.mean(true_y1 ** 2)))

    ccc = cc_num / (torch.mean(pred_y1 ** 2) + torch.mean(true_y1 ** 2) + torch.sum((ux - uy) ** 2))

    cc = cc.cpu().detach().numpy()
    ccc = ccc.cpu().detach().numpy()
    return ccc


class LSTMClassifier(nn.Module):

    def __init__(self, inp_dim, hidden_dim, batch_size, use_gpu):
        super(LSTMClassifier, self).__init__()
        module1 = nn.LSTM
        #module2 = nn.Linear
        self.layer1 = module1(
            inp_dim , hidden_dim,num_layers=2,bidirectional=False)
        #self.layer2 = module2(
        #    hidden_dim, out_dim)
        self.use_gpu = use_gpu
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.hidden = self.init_hidden()


    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, x):
        lstm_out, self.hidden =self.layer1(x)
        return lstm_out

class FCnn(nn.Module):

    def __init__(self, hidden_dim, out_dim):
        super(FCnn, self).__init__()
        module2 = nn.Linear
        self.layer2 = module2(
            hidden_dim, out_dim)

    def forward(self, x):
        out = torch.zeros_like(x[:, :, :1])
        for ii in range(x.shape[0]):
            out[ii,:,:] = torch.tanh(self.layer2(x[ii,:,:]))
        return out

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn

def delaycomp(data,delay,mode):

    #print(data[7511:7521,19])
    data = np.reshape(data, ( 9, 7501, data.shape[-1]))
    #print(data[1,11:21, 19])
    if mode == 0: # indicating the feature compensation
        data=data[:,:-delay,:]
    if mode == 1:
        data = data[:,delay:, :]

    dataf = []
    for i in range(data.shape[0]): # 9 utterences
        dataf.append(data[i,:,:])
    dataf = np.vstack(dataf)
    return dataf


################# Chunking data ########################################################################################
dim = 0
delaytime = [1]
name=['arousal','valence']
#seqs = [10,20,30,40,50]
seq=100

ftype=2
features=['mfcc','eGemaps','boaw']

Otype = dim
ODEname = ['alstm','vlstm']
ratios = [1.0]

filepath = 'path to data'
savepath = 'path to save model'

if __name__ == '__main__':
    batch_size = 100
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        ugpu = 1
    else:
        ugpu = 0

    for dd in range(len(delaytime)):
        delay = int(delaytime[dd] / 0.04)

        for ii in range(len(ratios)):
            ratio = ratios[ii]
            overlap = 0

            print("Loading festures..")
            data_all = io.loadmat(filepath + 'boaw_2s_cut.mat')
            X_train = data_all['data_train']
            X_test = data_all['data_dev']

            #X_train = delaycomp(X_train, delay, 0)

            print("Loading labels...")
            data_all = io.loadmat(filepath + 'goldstand_2s_cut.mat')
            Y_train = data_all['gt_train']
            Y_test = data_all['gt_dev']

            #Y_train = delaycomp(Y_train, delay, 1)


            ############## chunking data ####################
            X_train = chunking_frame(X_train,seq,overlap)
            X_test = chunking_frame(X_test,seq,overlap)
            Y_train = chunking_frame(Y_train, seq, overlap)
            Y_test = chunking_frame(Y_test, seq, overlap)

            X_train = np.swapaxes(X_train, 0, 1)
            X_test = np.swapaxes(X_test, 0, 1)
            Y_train = np.swapaxes(Y_train, 0, 1)
            Y_test = np.swapaxes(Y_test, 0, 1)

            Y_train = Y_train[:, :, dim]  # only taking the first dimension
            Y_test = Y_test[:,:, dim]

            ############## taking only partial data ################
            iid = int(np.floor(X_train.shape[1]/9))
            perm = torch.randperm(iid)
            idx1 = perm[:int(np.floor(iid * ratio))]
            idx=[]
            for i in range(9):
                idx = np.concatenate((idx,idx1+iid*i))
            idx = idx.astype(int)
            X_train = X_train[:, idx, :]
            Y_train = Y_train[:, idx]
            ############## taking only partial data ################

            #nhidden = X_train.shape[2]  # hidden neros of ODE layer
            obs_dim = 1  # output is 2 dimensions
            inp_dim = X_train.shape[2]  # input feature dimension
            batches_per_epoch = int(X_train.shape[1]/batch_size)

            ########## Initilizing matrix and parameters for network  #####################
            ccc_train = np.zeros((args.niters + 1, ))
            ccc_test = np.zeros((args.niters + 1, ))

            ccc_best = np.zeros(1, )

            ############## converting to tensors ##############################
            X_test = torch.from_numpy(X_test).float().to(device) # test data
            X_train= torch.from_numpy(X_train).float().to(device)  # train data
            Y_train = torch.from_numpy(Y_train).float().to(device) # train labels
            Y_test= torch.from_numpy(Y_test).float().to(device)  # test labels

            # model
            ODE_layer= [LSTMClassifier(args.inp_dim,args.hidden_dim,args.batch_size,ugpu)]
            model = nn.Sequential(*ODE_layer,*[FCnn(args.hidden_dim,obs_dim)]).to(device)
            params = (list(model.parameters()))
            lr_fn = learning_rate_with_decay(
                batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[10, 20, 30],
                decay_rates=[1, 0.1, 0.01]
            )
            optimizer = optim.Adam(params, lr=args.lr)
            loss_meter = RunningAverageMeter()

            for itr in range(1, args.niters + 1):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_fn(itr)

                optimizer.zero_grad()
                pred_train = model(X_train)

                # compute loss
                loss = CCCloss(torch.squeeze(pred_train),Y_train)
                loss.backward()
                optimizer.step()
                loss_meter.update(loss.item())

                print('Iter: {}, running avg elbo: {:.4f}'.format(itr, -loss_meter.avg))

                ccc_train[itr]=cc_cal(torch.squeeze(pred_train),Y_train)
                print('Train CCC is of ' + str(itr) + ' is ' + str(ccc_train[itr]))

                pred_test= model(X_test)
                pred_test1 = torch.reshape(torch.t(torch.squeeze(pred_test)), (-1,))
                lens = pred_test1.shape[0]

                p_temp = pred_test[0,0,0].repeat(int(delay))
                pred_test1 = torch.cat((p_temp, pred_test1), 0)
                pred_test1 = torch.unsqueeze(pred_test1[:lens],-1)
                pred_test = torch.from_numpy(chunking_frame(pred_test1.cpu().detach().numpy(),seq,overlap)).float().to(device)
                ccc_test[itr] = cc_cal(torch.squeeze(pred_test.permute(1,0,2)), Y_test)
                print('Test CCC is of ' + str(itr) + ' is ' + str(ccc_test[itr]))

                if ccc_test[itr] >ccc_best[0]:
                    ccc_best[0]=ccc_test[itr]
                    print('Best CCC is of ' + str(itr) + ' is ' + str(ccc_test[itr]))

                    p_train = pred_train.cpu().detach().numpy()
                    p_train=np.squeeze(p_train)
                    p_train=np.reshape(np.transpose(p_train),(p_train.shape[0]*p_train.shape[1],))
                    #p_train = sg.filtfilt(bcoef, 1, p_train) # no delay introduced as zero phase

                    p_test = pred_test.permute(1,0,2).cpu().detach().numpy()
                    p_test = np.squeeze(p_test)
                    p_test = np.reshape(np.transpose(p_test), (p_test.shape[0] * p_test.shape[1],))
                    #p_test = sg.filtfilt(bcoef, 1, p_test)  # no delay introduced as zero phase


                    #io.savemat('p_train_'+ ODEname[Otype] + str(seq) + name[dim]+ features[ftype] +str(delay) +'.mat', {"p_train": p_train})
                    io.savemat('p_test_' + ODEname[Otype] + str(ratio*100) + str(seq) + name[dim]+ features[ftype]  +str(delay) +'.mat', {"p_test": p_test})

            io.savemat('CCC_test_' + ODEname[Otype] + str(ratio*100)+ str(seq) + name[dim]+ features[ftype]  + str(delay) + '.mat', {"ccc_test": ccc_test})
            io.savemat('CCC_train_' + ODEname[Otype] + str(ratio*100) + str(seq) + name[dim]+ features[ftype]  +str(delay) + '.mat', {"ccc_train": ccc_train})

            del model