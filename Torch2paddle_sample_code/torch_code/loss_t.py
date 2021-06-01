import torch
import torch.nn as nn
from utils_t import calc_mean_std, mean_variance_norm

def calc_emd_loss(input, target):
    b,_,h,w=input.size()
    input = input.view(b, -1, w * h)
    input_norm = torch.sqrt((input**2).sum(1).view(b,-1,1))
    input = input.permute(0, 2, 1)
    target_t = target.view(b, -1, w*h)
    target_norm = torch.sqrt((target**2).sum(1).view(b,1,-1))
    similarity = torch.bmm(input, target_t)/input_norm/target_norm
    dist = 1.-similarity
    return dist

def calc_style_emd_loss(input, target):
    # emd loss
    CX_M = calc_emd_loss(input, target)
    m1, _ = CX_M.min(2)
    m2, _ = CX_M.min(1)
    loss_remd = torch.max(m1.mean(), m2.mean())
    return loss_remd

def calc_content_relt_loss(input, target):
    dM = 1.
    Mx = calc_emd_loss(input, input)
    Mx = Mx / Mx.sum(1, keepdim=True)
    My = calc_emd_loss(target, target)
    My = My / My.sum(1, keepdim=True)
    loss_content = torch.abs(dM * (Mx - My)).mean() * input.size(2) * input.size(3)
    return loss_content

def calc_content_loss(input, target, norm = False):
    mse_loss = nn.MSELoss()
    if(norm == False):
        return mse_loss(input, target)
    else:
        return mse_loss(mean_variance_norm(input), mean_variance_norm(target))

def calc_style_loss(input, target):
    mse_loss = nn.MSELoss()
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return mse_loss(input_mean, target_mean) + mse_loss(input_std, target_std)