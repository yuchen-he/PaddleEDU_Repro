import argparse
import os


class BaseOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):


        # Basic options
        self.parser.add_argument('--content_dir', type=str, default='/workspace/visCVPR2021/ZBK/data/coco/test1/',
                            help='Directory path to a batch of content images')
        self.parser.add_argument('--content_dir_test', type=str, default='/workspace/visCVPR2021/ZBK/data/coco/test1/',
                            help='Directory path to a batch of content images')
        self.parser.add_argument('--style_image', type=str, default='/workspace/visCVPR2021/ZBK/data/starrynew.png')

        # training options
        self.parser.add_argument('--save_dir', default='./experiments',
                            help='Directory to save the model')
        self.parser.add_argument('--log_dir', default='./logs',
                            help='Directory to save the log')
        self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--lr_decay', type=float, default=5e-5)
        self.parser.add_argument('--epoch', type=int, default=5)  
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--style_weight', type=float, default=3.0)
        self.parser.add_argument('--content_weight', type=float, default=1.0)
        self.parser.add_argument('--n_threads', type=int, default=16)
        self.parser.add_argument('--save_model_interval', type=int, default=1)
        self.parser.add_argument('--save_img_interval', type=int, default=1)
        self.parser.add_argument('--style_layers', default="r11,r21,r31,r41,r51", help='layers for style')
        self.parser.add_argument('--content_layers', default="r11,r21,r31,r41,r51", help='layers for content')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.content_layers = self.opt.content_layers.split(',')
        self.opt.style_layers = self.opt.style_layers.split(',')

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        path = "experiments/imgs"
        if not os.path.exists(path):
            os.makedirs(path)

        return self.opt
