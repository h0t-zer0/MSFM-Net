import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance


# # data augmentations
# def cv_random_flip(img, label):
#     flip_flag = random.randint(0, 1)
#     # flip_flag2= random.randint(0,1)
#     # left right flip
#     if flip_flag == 1:
#         img = img.transpose(Image.FLIP_LEFT_RIGHT)
#         label = label.transpose(Image.FLIP_LEFT_RIGHT)
#     return img, label


# def randomCrop(image, label):
#     border = 30
#     image_width = image.size[0]
#     image_height = image.size[1]
#     crop_win_width = np.random.randint(image_width - border, image_width)
#     crop_win_height = np.random.randint(image_height - border, image_height)
#     random_region = (
#         (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
#         (image_height + crop_win_height) >> 1)
#     return image.crop(random_region), label.crop(random_region)


# def randomRotation(image, label):
#     mode = Image.BICUBIC
#     if random.random() > 0.8:
#         random_angle = np.random.randint(-15, 15)
#         image = image.rotate(random_angle, mode)
#         label = label.rotate(random_angle, mode)
#     return image, labeldef cv_random_flip(rgb, ir, label):
def cv_random_flip(rgb, ir, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
        ir = ir.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return rgb, ir, label



def randomCrop(rgb, ir, label):
    border = 30
    image_width = rgb.size[0]
    image_height = rgb.size[1]

    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)

    random_region = (
        (image_width - crop_win_width) >> 1,
        (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1,
    )

    return rgb.crop(random_region), ir.crop(random_region), label.crop(random_region)


def randomRotation(rgb, ir, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        rgb = rgb.rotate(random_angle, mode)
        ir = ir.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return rgb, ir, label



def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


class CamObjDataset(data.Dataset):
    def __init__(self, rgb_root, ir_root, gt_root, trainsize, ir_transform="None"):
        self.trainsize = trainsize
        self.rgbs = [rgb_root + f for f in os.listdir(rgb_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.irs = [ir_root + f for f in os.listdir(ir_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.rgbs = sorted(self.rgbs)
        self.irs = sorted(self.irs)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.rgbs)
        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        assert ir_transform in ["None", "RGB", "IR"], "ir_transform must be 'None', 'RGB' or 'IR'"
        if ir_transform == "RGB":
            self.ir_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        elif ir_transform == "IR":
            self.ir_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.341,  0.360, 0.753], [0.208, 0.269, 0.241])])
        else:
            self.ir_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])


    def __getitem__(self, index):
        rgb = self.rgb_loader(self.rgbs[index])
        ir = self.rgb_loader(self.irs[index])
        gt = self.binary_loader(self.gts[index])

        rgb, ir, gt = cv_random_flip(rgb, ir, gt)
        rgb, ir, gt = randomCrop(rgb, ir, gt)
        rgb, ir, gt = randomRotation(rgb, ir, gt)
        rgb = colorEnhance(rgb)
        rgb = self.rgb_transform(rgb)
        ir = self.ir_transform(ir)
        gt = self.gt_transform(gt)

        return rgb, ir, gt

    def filter_files(self):
        assert len(self.rgbs) == len(self.irs) == len(self.gts), "The number of images and labels must be equal."
        rgbs, irs, gts = [], [], []
        for rgb_path, ir_path, gt_path in zip(self.rgbs, self.irs, self.gts):
            if rgb_path.split('/')[:-4] == ir_path.split('/')[:-4] == gt_path.split('/')[:-4]:
                rgbs.append(rgb_path)
                irs.append(ir_path)
                gts.append(gt_path)
        self.rgbs = rgbs
        self.irs = irs
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(rgb_root, ir_root, gt_root, batchsize, trainsize,
               shuffle=True, num_workers=4, pin_memory=True, ir_transform="None"):
    dataset = CamObjDataset(rgb_root, ir_root, gt_root, trainsize, ir_transform=ir_transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=True)
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, rgb_root, ir_root, gt_root, testsize, ir_transform="None"):
        self.testsize = testsize
        self.rgbs = [rgb_root + f for f in os.listdir(rgb_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.irs = [ir_root + f for f in os.listdir(ir_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]

        self.rgbs = sorted(self.rgbs)
        self.irs = sorted(self.irs)
        self.gts = sorted(self.gts)
        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        assert ir_transform in ["None", "RGB", "IR"], "ir_transform must be 'None', 'RGB' or 'IR'"
        if ir_transform == "RGB":
            self.ir_transform = transforms.Compose([
                transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        elif ir_transform == "IR":
            self.ir_transform = transforms.Compose([
                transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.341,  0.360, 0.753], [0.208, 0.269, 0.241])])
        else:
            self.ir_transform = transforms.Compose([
                transforms.Resize((self.testsize, self.testsize)),
                transforms.ToTensor()])
        self.gt_transform = transforms.ToTensor()

        self.size = len(self.rgbs)
        self.index = 0

    def load_data(self):
        rgb = self.rgb_loader(self.rgbs[self.index])
        ir = self.rgb_loader(self.irs[self.index])
        rgb = self.rgb_transform(rgb).unsqueeze(0)
        ir = self.ir_transform(ir).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])

        name = self.gts[self.index].split('/')[-1]

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return rgb, ir, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
