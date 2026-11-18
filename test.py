import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from Model import Network
from dataloader import test_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256)
parser.add_argument('--pth_path', type=str, default='../snapshot_64_v3_2backbone/TVCO1K/Net_epoch_best.pth')
parser.add_argument('--test_dataset_path', type=str, default='/root/autodl-tmp/TVCO1K/test')
parser.add_argument('--save_path', type=str, default='./snapshot_64_v3_2backbone/') 
parser.add_argument('--backbone', type=str, default='resnet101d')
parser.add_argument('--channels', type=int, default=64)

opt = parser.parse_args()

data_path = opt.test_dataset_path
save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)

# model = Network().cuda()
model = Network(
    channels=opt.channels,
    backbone=opt.backbone,
).cuda()

# model.load_state_dict(torch.load(opt.pth_path))

state_dict = torch.load(opt.pth_path, map_location='cpu')
msg = model.load_state_dict(state_dict, strict=False)
print(msg)

model.eval()

rgb_root = f'{data_path}/RGB/'
ir_root = f'{data_path}/T/'
gt_root = f'{data_path}/GT/'
test_loader = test_dataset(rgb_root, ir_root, gt_root, opt.testsize, ir_transform="RGB")

for i in range(test_loader.size):
    # rgb, gt, name = test_loader.load_data()
    rgb, ir, gt, name = test_loader.load_data()

        
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    rgb, ir = rgb.cuda(), ir.cuda()

    pred, _, _ = model(rgb, ir)

    res = F.interpolate(pred[0], size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    cv2.imwrite(save_path + name, res * 255)
