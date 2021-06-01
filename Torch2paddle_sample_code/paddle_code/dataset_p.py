import paddle
from paddle.vision import transforms
from paddle.io import Dataset
import os
from PIL import Image

class LapStyleDataset(Dataset):
    def __init__(self, content_root, style_root):
        super(LapStyleDataset, self).__init__()
        self.content_root = content_root
        self.paths = os.listdir(self.content_root)
        self.style_root = style_root
        self.transform = self.data_transform(128)

    def __getitem__(self, index):
        path = self.paths[index]
        content_img = Image.open(os.path.join(self.content_root,
                                               path)).convert('RGB')
        content_img = content_img.resize((128, 128), Image.BILINEAR)
        content_img = self.transform(content_img)
        style_img = Image.open(self.style_root).convert('RGB')
        style_img = style_img.resize((128, 128), Image.BILINEAR)
        style_img = self.transform(style_img)[:3, :, :]
        return content_img, style_img

    def data_transform(self, crop_size=128):
        transform_list = [
            transforms.RandomCrop(crop_size), 
            transforms.ToTensor()
            ] 
        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'LapStyleDataset'
