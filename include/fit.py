from torch.autograd import Variable
import torch
import torch.optim
import copy
import skimage.measure
import numpy as np
from scipy.linalg import hadamard
from scipy.stats import ortho_group
from .helpers import *
import torch.fft
import torch.nn.functional as F
import cv2
import os

if torch.cuda.device_count()==0:
    dtype = torch.FloatTensor
    device = 'cpu'
else:
    dtype = torch.cuda.FloatTensor
    device = 'cuda'

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=500, factor=0.5):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (factor**(epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print('\nLR is set to {}'.format(lr)) 
        print('\n')
    for param_group in optimizer.param_groups: 
        param_group['lr'] = lr
    return optimizer

def fit(net,
        num_channels, 
        img_clean_var,
        out_channels=1,
        m = 512, 
        d = 256, 
        net_input = None,
        decodetype='upsample',
        code = 'uniform', 
        opt_input = False, 
        find_best = False,
        OPTIMIZER ='adam',
        LR = 0.01, 
        numit_inner = 20,
        print_inner = False,
        optim = 'gd',
        OPTIMIZER2 = 'None',
        LR_LS = 0.02,
        num_iter = 5000,
        lr_decay_epoch = 0, 
        w = 0.0,
        rho = 1.0,
        ksi = 0.001,
        img_origin = None,
        weight_decay=0,
       ):
    
    if net_input is not None:
        net_input = net_input
        print(type(net_input))
        print("input provided")
    else:
        totalupsample = 2 ** (len(num_channels) - 1)
        width = int(d / (totalupsample))
        height = int(d / (totalupsample))
        shape = [1, num_channels[0], width, height]
        print("shape of latent code Z: ", shape)
        print("initializing latent code Z...")
        net_input = Variable(torch.zeros(shape)) 
        net_input.data.uniform_()
        net_input.data *= 1. / 10
    net_input_saved = net_input.data.detach()
    p = [t for t in net.decoder.parameters()]  # list of all weigths
    if (opt_input == True):  # optimizer over the input as well
        net_input.requires_grad = True
        print('optimizing over latent code Z1')
        p += [net_input]
    else:
        print('not optimizing over latent code Z1')

    mse_wrt_truth = np.zeros(num_iter)
    best_x = Variable(torch.zeros([1, out_channels, d, d]))
    # inner loop optimizer
    if OPTIMIZER == 'SGD':
        print("optimize decoder with SGD", LR)
        optimizer = torch.optim.SGD(p, lr=LR, momentum=0.9, weight_decay=weight_decay)
    elif OPTIMIZER == 'adam':
        print("optimize decoder with adam", LR)
        optimizer = torch.optim.Adam(p, lr=LR, weight_decay=weight_decay) # 优化器
    mse = torch.nn.MSELoss()
    if find_best:
        best_net = copy.deepcopy(net)
        best_mse = 1000000.0

    if optim == 'gd':
        print('optimizing with gradient descent...')
        x = net(net_input.type(dtype)).data.detach()
        for i in range(num_iter):
            if lr_decay_epoch is not 0:
                optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=lr_decay_epoch, factor=0.5)
            def closure():
                optimizer.zero_grad()
                outp = net(net_input.type(dtype))
                loss = mse(apply_f(outp, m), img_clean_var)
                loss.backward()
                mse_wrt_truth[i] = loss.data.cpu().numpy()
                return loss

            loss = optimizer.step(closure)
            print('Iteration %05d   Train loss %f ' % (i, loss.detach().cpu().numpy()), '\r', end='')
            if find_best:
                # if training loss improves by at least one percent, we found a new best net
                if best_mse > 1.01 * loss.detach().cpu().numpy():
                    best_mse = loss.detach().cpu().numpy()
                    best_net = copy.deepcopy(net)
        if find_best:
            net = best_net
    
    elif optim == 'pgd':
        print('optimizing with projected gradient descent...')
        x = Variable(torch.zeros([out_channels, d, d])).to(device)
        x.data = net(net_input.type(dtype))
        x_in = x.data.clone()
        x.requires_grad = True
        x.retain_grad()
        xvar = [x]
        if OPTIMIZER2 == 'SGD':
            optimizer2 = torch.optim.SGD(xvar, lr=LR_LS, momentum=0.9, weight_decay=weight_decay)
        elif OPTIMIZER2 == 'adam':
            optimizer2 = torch.optim.Adam(xvar, lr=LR_LS, weight_decay=weight_decay)

        for i in range(num_iter):
            if lr_decay_epoch is not 0:
                optimizer2 = exp_lr_scheduler(optimizer2, i, init_lr=LR_LS, lr_decay_epoch=lr_decay_epoch,
                                                factor=0.5)
                optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=lr_decay_epoch, factor=0.5)
            
            optimizer2.zero_grad()
            output = apply_f(x, m)
            loss_LS = mse(output, img_clean_var)
            loss_LS.backward()
            optimizer2.step()
            mse_wrt_truth[i] = loss_LS.item()
            print('Iteration %05d   Train loss %f ' % (i, mse_wrt_truth[i]), '\r', end='')
            for j in range(numit_inner):
                optimizer.zero_grad()
                out = net(net_input.type(dtype))
                loss_inner = mse(out, x)
                loss_inner.backward()
                optimizer.step()
                if print_inner:
                    print('Inner iteration %05d  Train loss %f' % (j, loss_inner.detach().cpu().numpy()))

            # project on learned network
            x.data = net(net_input.type(dtype))
            loss_updated = mse(apply_f(Variable(x.data, requires_grad=True), m), img_clean_var)
            if find_best:
                # if training loss improves by at least one percent, we found a new best net
                if best_mse > 1.01 * loss_updated.item():
                    best_mse = loss_updated.item()
                    best_net = copy.deepcopy(net)
                    best_x = x

        if find_best:
            net = best_net

    elif optim == 'admm':
        print('optimizing with ADMM...')
        I = torch.ones((1, out_channels, m, m)).to(device)
        W = w*I
        b = img_clean_var
        v = Variable(torch.zeros([1, out_channels, m, m]))
        x = Variable(torch.zeros([1, out_channels, d, d]))
        x = x.to(device)
        v = ifftn(b).to(device)

        for i in range(num_iter):
            if lr_decay_epoch is not 0:
                optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=lr_decay_epoch, factor=0.5)
            x = F.pad(v-W / rho, (0, d-m, 0, d-m), "constant", 0)
            temp2 = fftn(F.pad(x, (0, m - d, 0, m - d), "constant", 0) + W / rho, m)
            temp3 = torch.abs(temp2)
            temp1 = torch.sqrt(b ** 2 + ksi * I) / torch.sqrt(temp3 ** 2 + ksi * I)
            v = ifftn(temp1 * temp2)
            # project on learned network
            for j in range(numit_inner):
                optimizer.zero_grad()
                out = net(net_input.type(dtype))
                loss_inner = mse(out, v[:, :, 0:d, 0:d])
                loss_inner.backward()
                optimizer.step()
                if print_inner:
                    print('Inner iteration %05d  Train loss %f' % (j, loss_inner.detach().cpu().numpy()))
            v.data[:, :, 0:d, 0:d] = net(net_input.type(dtype))
            W = W + rho * (F.pad(x, (0, m - d, 0, m - d), "constant", 0) - v)
            output = apply_f(x, m)
            loss_LS = mse(output, img_clean_var)
            mse_wrt_truth[i] = loss_LS.item()
            print('Iteration %05d   Train loss %f ' % (i, mse_wrt_truth[i]), '\r', end='')

            loss_updated = mse(output, img_clean_var)
            if find_best:
                # if training loss improves by at least one percent, we found a new best net
                if best_mse > 1.01 * loss_updated.item():
                    best_mse = loss_updated.item()
                    best_net = copy.deepcopy(net)
                    best_x = x
        if find_best:
            net = best_net

    return mse_wrt_truth, net_input_saved, net, net_input, best_x