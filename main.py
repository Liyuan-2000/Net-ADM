from __future__ import print_function
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
from include import *
from PIL import Image
import PIL
import pywt
import numpy as np
import torch.nn.functional as F
import torch
import torch.optim
from torch.autograd import Variable
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.color import rgb2ycbcr
from sklearn import linear_model
import torch.fft
import cv2
import csv

GPU = True
if GPU == True:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    print("num GPUs",torch.cuda.device_count())
    device = 'cuda'
    if torch.cuda.device_count()==0:
        dtype = torch.FloatTensor
        device = 'cpu'
else:
    dtype = torch.FloatTensor
    device = 'cpu'
import time
from skimage.util import random_noise

def add_noise(b, SNR):
    if b.shape[1] == 1:
        sigma = torch.sqrt(torch.var(b)/(10**(SNR/20)))
        noise = np_to_var(np.random.normal(loc=0.0, scale=sigma.data.cpu().numpy(), size=(1, m, m))).type(dtype)
    elif b.shape[1] == 3:
        (b_r, b_g, b_b) = torch.split(b, [1, 1, 1], dim=1)
        sigma_r, sigma_g, sigma_b = torch.sqrt(torch.var(b_r)/(10**(SNR/20))), \
                                    torch.sqrt(torch.var(b_g) / (10**(SNR/20))), \
                                    torch.sqrt(torch.var(b_b) / (10**(SNR/20)))
        noise = torch.cat((np_to_var(np.random.normal(loc=0.0, scale=sigma_r.data.cpu().numpy(), size=(1, m, m))).type(dtype),\
                           np_to_var(np.random.normal(loc=0.0, scale=sigma_g.data.cpu().numpy(), size=(1, m, m))).type(dtype),\
                           np_to_var(np.random.normal(loc=0.0, scale=sigma_b.data.cpu().numpy(), size=(1, m, m))).type(dtype)), 1)
    noisy_b = b + noise
    return noisy_b

# record PSNR and SSIM
data = open('data.csv', 'w', encoding = 'utf8', newline = '')
writer = csv.writer(data)

dataset = 'celeba' # 'mnist', 'Cameraman'
filename = 'celeba1.png'
ff = [2.0, 1.8] # sampling rate
circ_num = 10 # number of runs at each sampling rate
optim = 'admm' # 'gd', 'pgd'
is_noise = False # wether to add noise
SNR = 70 # noise level, if is_noise = True
weight_decay = 0.00001

if os.path.exists('output') == False:
    os.mkdir('output')
if os.path.exists('output/' + optim) == False:
    os.mkdir('output/' + optim)

for i in range(len(ff)):
    max_psnr = 0
    for j in range(circ_num):
        img_path = os.path.join('images/' + dataset + '/',filename)
        f = ff[i]
        print('sampling rate: ', f)
        img_pil = Image.open(img_path)
        # number of channels in each layer of network
        if dataset == 'mnist':
            num_channels = [25, 15, 10]
        elif dataset == 'celeba':
            num_channels = [120, 25, 15, 10]
        elif dataset == 'Cameraman':
            num_channels = [128, 64, 64, 32]

        img_np = pil_to_np(img_pil) # W x H x C [0...255] to C x W x H [0...1]
        img_np_orig = 1 * img_np
        print('Dimensions of input image:', img_np.shape)

        output_depth = img_np.shape[0]
        d = img_np.shape[1]
        img_var = np_to_var(img_np).type(dtype)
        m = int(f * d)
        print('number of measurement:', m)
        img_var_meas = np_to_var(np.zeros((output_depth, m, m))).type(dtype)
        img_var_meas1 = apply_f(img_var, m)
        if is_noise:
            img_var_meas = add_noise(img_var_meas1, SNR)
        else:
            img_var_meas = img_var_meas1
       
        net = autoencodernet(num_output_channels=output_depth, num_channels_up=num_channels, need_sigmoid=True,
                             decodetype='upsample'
                             ).type(dtype)
        # print("number of parameters: ", num_param(net))
        # print(net.decoder)
        net_in = copy.deepcopy(net)

        if optim == 'gd':
            OPTIMIZER = 'adam'  # optimizer - SGD or adam
            numit = 5000  # number of iterations for SGD or adam
            LR = 0.005  # required for gd 

            OPTIMIZER2 = None
            numit_inner = None
            LR_LS = None

            lr_decay_epoch = 2500 # decay learning rates of optimizers
            img_origin = img_np_orig
            
        elif optim == 'pgd':
            OPTIMIZER2 = 'adam'  # outer loop optimizer - SGD or adam
            numit = 1000  # number of outer iterations of LS
            LR_LS = 0.5  # required for outer loop of LS

            OPTIMIZER = 'adam'  # inner loop optimizer - SGD or adam
            numit_inner = 5  # number of inner loop iterations for projection
            LR = 0.0005  # required for pgd/inner loop of projection

            lr_decay_epoch = 500  # decay learning rates of both inner and outer optimizers
            img_origin = img_np_orig

        elif optim == 'admm':
            OPTIMIZER = 'adam' # inner loop optimizer - SGD or adam
            numit = 1000 # number of iterations for SGD or adam
            numit_inner = 5 # number of inner loop iterations for projection
            LR = 0.005  # required for inner loop of projection

            OPTIMIZER2 = None
            LR_LS = None

            lr_decay_epoch = 500 # decay learning rates of outer optimizers
            img_origin = img_np_orig

        decodetype = 'upsample'
        t0 = time.time()
        mse_t, ni, net, ni_mod, in_np_img = fit(
            net=net,
            num_channels=num_channels,
            m=m,
            d=d,
            num_iter=numit,
            numit_inner=numit_inner,
            LR=LR,
            LR_LS=LR_LS,
            OPTIMIZER=OPTIMIZER,
            OPTIMIZER2=OPTIMIZER2,
            lr_decay_epoch=lr_decay_epoch,
            img_clean_var=img_var_meas,
            find_best=True,
            code='uniform',
            weight_decay=weight_decay,
            decodetype=decodetype,
            optim=optim,
            out_channels=output_depth,
            img_origin=img_origin
        )
        t1 = time.time()
        print('\ntime elapsed:', t1 - t0)
        if optim == 'admm':
            out_img_np = in_np_img.data.cpu().numpy()[0]
            out_img_np = convert(out_img_np)
        else:
            out_img_np = net(ni.type(dtype)).data.cpu().numpy()[0]
            out_img_np = convert(out_img_np)

        img_path2 = os.path.join('images/' + dataset + '_/',filename)
        if output_depth == 1:
            img_np1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_np2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
            out_img_np = out_img_np[0]
            SSIM1 = ssim(out_img_np, img_np1)
            SSIM2 = ssim(out_img_np, img_np2)
        else:
            img_np1 = cv2.imread(img_path)
            img_np2 = cv2.imread(img_path2)
            out_img_np = out_img_np.transpose(1, 2, 0)
            out_img_np = out_img_np[:, :, ::-1]
            SSIM1 = ssim(out_img_np, img_np1, multichannel=True)
            SSIM2 = ssim(out_img_np, img_np2, multichannel=True)
        PSNR1 = psnr(out_img_np, img_np1)
        PSNR2 = psnr(out_img_np, img_np2)
        writer.writerow([filename, str(f), str(max(PSNR1, PSNR2)), str(max(SSIM1, SSIM2))])
        print('iter = ', j)
        print('psnr = ', max(PSNR1, PSNR2))
        print('ssim = ', max(SSIM1, SSIM2))
        if max(PSNR1, PSNR2) > max_psnr:
            if is_noise:
                print(os.path.exists('output/' + optim + '/' + 'noise'))
                if os.path.exists('output/' + optim + '/' + 'noise') == False:
                    os.mkdir('output/' + optim + '/' + 'noise')
                out_path = os.path.join('output/' + optim + '/' + 'noise', str(f) + '_' +str(SNR) + filename)
            else:
                print(os.path.exists('output/' + optim + '/' + filename[0:-4]))
                if os.path.exists('output/' + optim + '/' + filename[0:-4]) == False:
                    os.mkdir('output/' + optim + '/' + filename[0:-4])
                out_path = os.path.join('output/' + optim + '/' + filename[0:-4], str(f) + filename)
            cv2.imwrite(out_path, out_img_np)
            max_psnr = max(PSNR1, PSNR2)
data.close()
print("over!")
