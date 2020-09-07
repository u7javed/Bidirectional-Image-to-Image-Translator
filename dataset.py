import torch
import torchvision.transforms as transforms
from PIL import Image
from glob import glob

class Image2ImageDataSet(torch.utils.data.Dataset):
    def __init__(self, seg_dir, real_dir, width, height, sample_size=1000):
        self.transformation = transforms.Compose([
            transforms.Resize((width, height), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])

   
        seg_files = glob(seg_dir + '*.png')
        real_files = glob(real_dir + '*.jpg')

        self.seg_list = []
        self.real_list = []

        for image in range(sample_size):
            self.seg_list.append(Image.open(seg_files[image]))
            self.real_list.append(Image.open(real_files[image]))

    def __len__(self):
        return len(self.seg_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seg_image = self.seg_list[idx]
        real_image = self.real_list[idx]

        seg_image = self.transformation(seg_image)
        real_image = self.transformation(real_image)

        return seg_image, real_image