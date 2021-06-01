import paddle
from paddle.io import DataLoader
from PIL import Image
from PIL import ImageFile
import numpy as np

from utils_p import save_params
from dataset_p import LapStyleDataset
from net_p import Encoder, DecoderNet
from loss_p import calc_style_emd_loss, calc_content_loss, calc_content_relt_loss, calc_style_loss
from option_p import BaseOptions

opt = BaseOptions().parse()
Image.MAX_IMAGE_PIXELS = None  
ImageFile.LOAD_TRUNCATED_IMAGES = True  


def inference(encoder, decoder):
    decoder.eval()
    encoder.eval()

    with paddle.no_grad():
        for content, style in test_iter:
            cF = encoder(content)
            sF = encoder(style)
            content = decoder(cF, sF) 
    decoder.train()
    encoder.train()  

def train(content, style, net_enc, net_dec, opt):
    sF = net_enc(style)
    cF = net_enc(content)
    stylized = net_dec(cF, sF)
    tF = net_enc(stylized)

    """content loss"""
    loss_c = 0
    for layer in opt.content_layers:
        loss_c += calc_content_loss(tF[layer], cF[layer], norm = True)

    """style loss"""
    loss_s = 0
    for layer in opt.style_layers:
        loss_s += calc_style_loss(tF[layer], sF[layer])

    """IDENTITY LOSSES"""
    Icc = net_dec(cF, cF)
    l_identity1 = calc_content_loss(Icc, content)
    Fcc = net_enc(Icc)
    l_identity2 = calc_content_loss(Fcc['r11'], cF['r11'])
    for layer in ['r21', 'r31', 'r41', 'r51']:
        l_identity2 += calc_content_loss(Fcc[layer], cF[layer])

    """relative loss"""
    loss_style_remd = calc_style_emd_loss(tF['r31'], sF['r31']) + calc_style_emd_loss(tF['r41'], sF['r41'])
    loss_content_relt = calc_content_relt_loss(tF['r31'], cF['r31']) + calc_content_relt_loss(tF['r41'], cF['r41'])

    loss_c = opt.content_weight * loss_c
    loss_s = opt.style_weight * loss_s
    loss = loss_c + loss_s + l_identity1 * 50 + l_identity2 * 1 + loss_style_remd * 10 + loss_content_relt * 16
    return loss

# define model and optimizer
net_dec = DecoderNet()
net_dec.set_dict(paddle.load('/workspace/visCVPR2021/ZBK/pre_trained/decoder_iter_10000.pdparams'))
net_enc = Encoder()

optimizer = paddle.optimizer.Adam(learning_rate=opt.lr, parameters=net_dec.parameters())

# define dataload
train_dataset = LapStyleDataset(opt.content_dir, opt.style_image)
test_dataset = LapStyleDataset(opt.content_dir_test, opt.style_image)

train_iter = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.n_threads)
test_iter = DataLoader(test_dataset, batch_size=1, num_workers=0)
                                         
# training 
for epoch in range(opt.epoch):
    for i, item in enumerate(train_iter):
        content_image = item[0]
        style_image = item[1]
        optimizer.clear_grad()
        loss = train(content_image, style_image, net_enc, net_dec, opt)
        print('epoch:', epoch, 'iter:', i, 'loss:%.2f'% np.array(loss))
        loss.backward()
        optimizer.step()
        # for i, param in enumerate(net_dec.named_parameters()):
        #     if i>0:
        #         break
        #     print(param[0], param[1], param[1].grad)

    if (epoch + 1) % opt.save_img_interval == 0:
        inference(net_enc, net_dec)

    if (epoch + 1) % opt.save_model_interval == 0 or (epoch + 1) == opt.epoch:
        save_params(net_dec, "decoder", opt.save_dir, epoch+1)