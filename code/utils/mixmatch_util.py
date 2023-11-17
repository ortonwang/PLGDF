# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import torch
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
import random





def augmentation_torch(volume, aug_factor):
    # volume is numpy array of shape (C, D, H, W)
    noise = torch.clip(torch.randn(*volume.shape) * 0.1, -0.2, 0.2).cuda()
    return volume + aug_factor * noise#.astype(np.float32)

def mix_module(X, U, eval_net, K, T, alpha, mixup_mode, aug_factor):
    X_b = len(X)
    U_b = len(U)

    # step 1: Augmentation with random noise
    # aug_factor = torch.tensor(aug_factor)
    X_cap = [(augmentation_torch(x[0], aug_factor), x[1]) for x in X]

    U_cap = U.repeat(K, 1, 1, 1, 1)  # [K*b, 1, D, H, W]

    U_cap += torch.clamp(torch.randn_like(U_cap) * 0.1, -0.2, 0.2)  # augmented.

    # step 2: label guessing
    with torch.no_grad():
        Y_u,_,_,_ = eval_net(U_cap)  #the model now is four output, and the teacher network only use the first output
        Y_u = F.softmax(Y_u, dim=1)
        # print(Y_u.shape)
    guessed = torch.zeros(U.shape).repeat(1, K, 1, 1, 1)
    # if GPU:
    guessed = guessed.cuda()
    for i in range(K):
        guessed += Y_u[i * U_b:(i + 1) * U_b]
    guessed /= K  # to get the two experience average result

    guessed = guessed.repeat(K, 1, 1, 1, 1)
    guessed = torch.argmax(guessed, dim=1)
    pseudo_label = guessed      #get the pseudo label

    U_cap = list(zip(U_cap, guessed))  # Merge pseudo labels and augumented data

    ## Now we have X_cap ,list of (data, label) of length b, U_cap, list of (data, guessed_label) of length k*b

    # step 3: MixUp
    # original paper mathod

    x_mixup_mode, u_mixup_mode = mixup_mode[0], mixup_mode[1]

    if x_mixup_mode == '_':
        X_prime = X_cap
    else:
        raise ValueError('wrong mixup_mode')

    if u_mixup_mode == 'x':
        idxs = np.random.permutation(range(U_b * K)) % X_b
        U_prime = [mix_up(U_cap[i], X_cap[idxs[i]], alpha) for i in range(U_b * K)]
    else:
        raise ValueError('wrong mixup_mode')

    return X_prime, U_prime, pseudo_label

def mix_up(s1, s2, alpha):
    # print('??????', s1[0].shape, s1[1].shape, s2[0].shape, s2[1].shape)
    # s1, s2 are tuples(data, label)
    l = np.random.beta(alpha, alpha)  # 原文公式(8)
    l = max(l, 1 - l)  # 原文公式(9)

    x1, p1 = s1
    x2, p2 = s2

    x = l * x1 + (1 - l) * x2  # 原文公式(10)
    p = l * p1 + (1 - l) * p2  # 原文公式(11)

    return (x, p)


