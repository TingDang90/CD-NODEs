import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as io
from torch.utils.data import Dataset, DataLoader, TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=True)
parser.add_argument('--visualize', type=eval, default=False)
#parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--nepochs', type=int, default=60)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--rtol', type=float, default=1e-7)
parser.add_argument('--atol', type=float, default=1e-13)
parser.add_argument('--PATIENCE', type=int, default=30)
parser.add_argument('--latent_dim', type=int, default=64)
parser.add_argument('--nhidden', type=int, default=64)
parser.add_argument('--kk',type=int,default=10)
parser.add_argument('--seq', type=int, default=100)
parser.add_argument('--scale',type=int,default = 1)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


def chunking_frame(data, window, overlap):
    shift = window - overlap
    frame_num = int(np.ceil(data.shape[0] / float(shift)))
    chunked_data = np.zeros((frame_num, window, data.shape[1]), dtype='float')
    i_w = 0
    for wn in range(frame_num):
        if (wn == frame_num - 1):
            chunked_data[wn, 0:np.shape(data[i_w:, :])[0], :] = data[i_w:, :]
        else:
            chunked_data[wn, 0:window, :] = data[i_w:i_w + window, :]
        i_w = i_w + shift
    return chunked_data


def Binomialfilter(delay_frame):
    h = [0.5, 0.5]
    j = int(2 * delay_frame + 1)
    binomialcoff = np.convolve(h, h)
    for i in range(j - 3):
        binomialcoff = np.convolve(binomialcoff, h)
    binomialcoff = binomialcoff / (np.sum(binomialcoff))
    return binomialcoff

def CCCloss(pred_y, true_y):
    ux = torch.mean(pred_y)
    uy = torch.mean(true_y)
    pred_y1 = pred_y - ux
    true_y1 = true_y - uy
    cc = torch.sum(pred_y1 * true_y1) / (
            torch.sqrt(torch.sum(pred_y1 ** 2)) * torch.sqrt(torch.sum(true_y1 ** 2)))
    # print(cc.device)

    if device.type == 'cpu':
        cc_num = torch.tensor(2.0) * cc * (
                torch.sqrt(torch.mean(pred_y1 ** 2)) * torch.sqrt(torch.mean(true_y1 ** 2))).to(device.type)
        ccc = cc_num / (torch.mean(pred_y1 ** 2) + torch.mean(true_y1 ** 2) + torch.sum((ux - uy) ** 2))
        ccc = torch.tensor(1.0) - ccc
    else:
        cc_num = torch.cuda.FloatTensor(1, ).fill_(2.0) * cc * (
                torch.sqrt(torch.mean(pred_y1 ** 2)) * torch.sqrt(torch.mean(true_y1 ** 2))).to(device)
        ccc = cc_num / (torch.mean(pred_y1 ** 2) + torch.mean(true_y1 ** 2) + torch.sum((ux - uy) ** 2))
        ccc = torch.cuda.FloatTensor(1, ).fill_(1.0) - ccc
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


class PJClassifier(nn.Module):# projection to new space

    def __init__(self, inp_dim, latent_dim, use_gpu):
        super(PJClassifier, self).__init__()

        self.layer1 = nn.Linear(inp_dim, latent_dim)
        #self.layer2 = nn.Linear(latent_dim, 1)

        self.use_gpu = use_gpu
        self.latent_dim = latent_dim

    def forward(self, x):
        out = self.layer1(x)
        out = torch.tanh(out) # this out is the embeddings
        #ini= self.layer2(out[0,:]) # this out is initial values
        return out

class initpred(nn.Module):# projection to new space

    def __init__(self, latent_dim, out_dim, use_gpu):
        super(initpred, self).__init__()

        self.layer1 = nn.Linear(latent_dim, out_dim)
        self.use_gpu = use_gpu

    def forward(self, x):
        out = self.layer1(x)
        out = torch.tanh(out) # this out is the initial values
        return out


class LatentODEfunc(nn.Module):

    def __init__(self, inp_dim, nhidden, out_dim):
        super(LatentODEfunc, self).__init__()
        #self.elu = nn.tanh(inplace=True)
        self.fc1 = nn.Linear(inp_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, out_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = torch.tanh(out)
        out = self.fc2(out)
        out = torch.tanh(out)
        out = self.fc3(out)
        #out = args.scale * torch.tanh(out/args.scale)
        #out = (torch.tanh(out)+1.0)/2.0*2.5-0.9583
        #out = (torch.tanh(out) + 1.0) / 2.0 * 4 - 2
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


################# Chunking data ########################################################################################
dim = 0
delay = 50
ts_size = 7501
batch_size = int(ts_size) - int(delay)

out_dim = 1 #equal to the output dimension
latent_dim = args.latent_dim
nhidden = args.nhidden
bcoef = Binomialfilter(delay)


name = [str(nhidden) + 'artanh'+str(args.scale),str(nhidden) + 'vrtanh'+str(args.scale)]

ftype = 2
features = ['mfcc', 'eGemaps', 'boaw']

filepath = '/Users/sally/Documents/ODE/BoAW/'
savepath = '/Users/sally/Documents/ODE/BoAW'

if __name__ == '__main__':


    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        ugpu = 0
    else:
        ugpu = 1

    seq = args.seq
    overlap = 0

    ################# lOAD DATA ############################################################################################
    print("Loading festures..")
    data_all = io.loadmat(filepath + 'boaw_2s_cut.mat')
    X_train = data_all['data_train']
    X_test = data_all['data_dev']

    ################# lOAD DATA ############################################################################################
    print("Loading labels...")
    data_all = io.loadmat(filepath + 'goldstand_2s_cut.mat')
    Y_train = data_all['gt_train']
    Y_test = data_all['gt_dev']

    Y_train = Y_train[:,dim]
    Y_test = Y_test[:,dim]

    inp_dim = X_train.shape[1]  # input feature dimension


    ########## Initilizing matrix and parameters for network  #####################
    ccc_train = np.zeros((args.nepochs + 1,))
    ccc_test = np.zeros((args.nepochs + 1,))
    ccc_best = -1000 * np.ones(1, )

    ############## Forming pytorch dataset ###############
    X_train = torch.from_numpy(X_train).float() # train data
    Y_train = torch.from_numpy(Y_train).float() # train labels

    X_test = torch.from_numpy(X_test).float()  # train data
    Y_test = torch.from_numpy(Y_test).float()  # train labels

    train_dataset = TensorDataset(X_train,Y_train)
    test_dataset = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=ts_size*9,
                              shuffle=False)
    batches_per_epoch = len(train_loader)

    #### model
    rec = PJClassifier(inp_dim, latent_dim, ugpu).to(device)
    pri = initpred(latent_dim,out_dim,ugpu).to(device)
    func = LatentODEfunc(latent_dim + out_dim, nhidden, out_dim).to(device)
    params = (list(rec.parameters()) +  list(pri.parameters()) + list(func.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

    loss_meter = RunningAverageMeter()

    for itr in range(args.nepochs+1): # epoch number
        for i,data in enumerate(train_loader):
            print('Epoch = ' + str(itr) + ' Iteration ='+ str(i))
            inputs,labels = data
            inputs = rec.forward(inputs.float().to(device))  # converting to low dimensions of all features

            inputs = torch.from_numpy(chunking_frame(inputs.cpu().detach().numpy(), seq, overlap)).permute(1,0,2)
            labels_new = torch.from_numpy(chunking_frame(torch.unsqueeze(labels,1).cpu().detach().numpy(), seq, overlap)).permute(1,0,2)

            inputs = inputs.float().to(device)  # train data
            labels = labels.float().to(device)  # train labels
            print(inputs.shape)
            optimizer.zero_grad()
            ############################ Start trainiing the model #######

            ts = torch.linspace(0, (seq-1)*0.04, steps=seq)
            y0_new = pri.forward(inputs[0,:,:]).float().to(device)
            #y0_new = labels_new[0,:,:].float().to(device)
            pred_train = odeint(func, y0_new, ts, inputs, args.rtol,args.atol)

            ################## plot #####################
            pred_train = torch.reshape(torch.t(pred_train[:, :, 0]), (-1,))
            print(pred_train.shape)
            print(labels.shape)
            loss = CCCloss(pred_train[:batch_size],labels)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            print('Iter: {}, running avg elbo: {:.4f}'.format(itr, -loss_meter.avg))

            #################################################################
            if i == 0:
                p_ts = pred_train[:batch_size]
            else:
                p_ts = torch.cat((p_ts, pred_train[:batch_size]))

        ccc_train[itr] = cc_cal(p_ts, Y_train.to(device))
        print('Train CCC is of ' + str(itr) + ' is ' + str(ccc_train[itr]))

        scheduler.step()

        with torch.no_grad(): ### after each epoch training
            ######################## Test may be able to be changed
            for i, data in enumerate(test_loader):# for each utterence
                te_fea, te_lab = data
                te_fea = rec.forward(te_fea.float().to(device))  # converting to low dimensions of all features

                te_fea=chunking_frame(te_fea.cpu().detach().numpy(), ts_size, 0)
                te_fea=torch.from_numpy(te_fea).float().to(device)
                te_fea = te_fea.permute(1, 0, 2)

                tt = torch.linspace(0, (te_fea.shape[0] - 1) * 0.04, steps=te_fea.shape[0])
                te_lab1 = pri.forward(te_fea[0,:,:]).float().to(device)
                p_test = odeint(func, te_lab1, tt, te_fea, args.rtol, args.atol)

                ################## post processing for delay per utterence ##########
                p_test = p_test[:ts_size,:,:]
                p_temp = p_test[0,:,:].repeat(int(delay),1,1)
                p_test = torch.cat((p_temp,p_test[:-int(delay) + 1,:,:]),0)

                #################################################################
                if i == 0:
                    p_ts = torch.reshape(torch.t(p_test[:ts_size,:,0]),(-1,))
                else:
                    p_ts = torch.cat((p_ts, torch.reshape(torch.t(p_test[:ts_size,:,0]),(-1,))))

            ccc_test[itr] = cc_cal(p_ts,Y_test.to(device))
            print('Test CCC is of ' + str(itr) + ' is ' + str(ccc_test[itr]))

            if ccc_test[itr] > ccc_best[0]:
                ccc_best[0] = ccc_test[itr]
                curr_step = 0
                print('CCC best = ' + str(ccc_best[0]) + ' epoch = ' + str(itr))
                io.savemat(
                    savepath + 'p_tsfd_' + str(args.kk) + '_' + str(seq) + name[dim] + features[ftype] + '.mat',
                    {"p_test": p_ts.cpu().detach().numpy()})
                torch.save(rec.state_dict(), 'rec' + str(args.kk) + '_' + str(seq) + name[dim])
                torch.save(pri.state_dict(), 'pri' + str(args.kk) + '_' + str(seq) + name[dim])
                torch.save(func.state_dict(), 'func' + str(args.kk) + '_' + str(seq) + name[dim])
            else:
                curr_step += 1
                if curr_step == args.PATIENCE:
                    print('Early Sopp!')
                    break

    io.savemat( savepath + 'CCC_tests_' + str(args.kk) + '_' + str(seq) + name[dim] + features[ftype] + '.mat',{"ccc_test": ccc_test})
    io.savemat(savepath + 'CCC_trains_' + str(args.kk) + '_' + str(seq) + name[dim] + features[ftype] + '.mat',{"ccc_train": ccc_train})

    del train_dataset
    del test_dataset
    del train_loader
    del test_loader
    del rec, pri, func, params
    torch.cuda.empty_cache()
