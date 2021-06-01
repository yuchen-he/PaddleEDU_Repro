import paddle
import paddle.nn as nn
from utils_p import calc_mean_std, mean_variance_norm

def calc_emd_loss(pred, target):
    b, _, h, w = pred.shape
    pred = pred.reshape([b, -1, w * h])
    pred_norm = paddle.sqrt((pred**2).sum(1).reshape([b, -1, 1]))
    pred = pred.transpose([0, 2, 1])
    target_t = target.reshape([b, -1, w * h])
    target_norm = paddle.sqrt((target**2).sum(1).reshape([b, 1, -1]))
    similarity = paddle.bmm(pred, target_t) / pred_norm / target_norm
    dist = 1. - similarity
    return dist

def calc_style_emd_loss(pred, target):
    CX_M = calc_emd_loss(pred, target)
    m1 = CX_M.min(2)
    m2 = CX_M.min(1)
    m = paddle.concat([m1.mean(), m2.mean()])
    loss_remd = paddle.max(m)
    return loss_remd

def calc_content_relt_loss(pred, target):
    dM = 1.
    Mx = calc_emd_loss(pred, pred)
    Mx = Mx / Mx.sum(1, keepdim=True)
    My = calc_emd_loss(target, target)
    My = My / My.sum(1, keepdim=True)
    loss_content = paddle.abs(
        dM * (Mx - My)).mean() * pred.shape[2] * pred.shape[3]
    return loss_content

def calc_content_loss(pred, target, norm=False):
    mse_loss = nn.MSELoss()
    if (norm == False):
        return mse_loss(pred, target)
    else:
        return mse_loss(mean_variance_norm(pred), mean_variance_norm(target))

def calc_style_loss(pred, target):
    mse_loss = nn.MSELoss()
    pred_mean, pred_std = calc_mean_std(pred)
    target_mean, target_std = calc_mean_std(target)
    return mse_loss(pred_mean, target_mean) + mse_loss(pred_std, target_std)

# class calc_style_emd_loss():
#     def __init__(self):
#         super(calc_style_emd_loss, self).__init__()

#     def __call__(self, pred, target):
#         CX_M = calc_emd_loss(pred, target)
#         m1 = CX_M.min(2)
#         m2 = CX_M.min(1)
#         m = paddle.concat([m1.mean(), m2.mean()])
#         loss_remd = paddle.max(m)
#         return loss_remd

# class calc_content_relt_loss():
#     def __init__(self):
#         super(calc_content_relt_loss, self).__init__()

#     def __call__(self, pred, target):
#         dM = 1.
#         Mx = calc_emd_loss(pred, pred)
#         Mx = Mx / Mx.sum(1, keepdim=True)
#         My = calc_emd_loss(target, target)
#         My = My / My.sum(1, keepdim=True)
#         loss_content = paddle.abs(
#             dM * (Mx - My)).mean() * pred.shape[2] * pred.shape[3]
#         return loss_content

# class calc_content_loss():
#     def __init__(self):
#         self.mse_loss = nn.MSELoss()

#     def __call__(self, pred, target, norm=False):
#         if (norm == False):
#             return self.mse_loss(pred, target)
#         else:
#             return self.mse_loss(mean_variance_norm(pred),
#                                  mean_variance_norm(target))

# class calc_style_loss():
#     def __init__(self):
#         self.mse_loss = nn.MSELoss()

#     def __call__(self, pred, target):
#         pred_mean, pred_std = calc_mean_std(pred)
#         target_mean, target_std = calc_mean_std(target)
#         return self.mse_loss(pred_mean, target_mean) + self.mse_loss(
#             pred_std, target_std)
