import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from Model import Network
from dataloader import get_loader, test_dataset
import logging
import torch.backends.cudnn as cudnn
from torch import optim
from mssim import ssim


def iou(prob, gt):
    inter = torch.sum(gt * prob, dim=(1, 2, 3))
    union = gt.sum(dim=(1, 2, 3)) + prob.sum(dim=(1, 2, 3)) - inter
    iou = inter / union
    return iou.mean()

def loss_func(logits, seg_gts):
    losses = []
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(input=logits, target=seg_gts, reduction="mean")
    losses.append(bce_loss)
    prob = logits.sigmoid()
    ssim_loss = 1 - ssim(prob, seg_gts)
    losses.append(ssim_loss)
    iou_loss1 = 1 - iou(prob, seg_gts)
    losses.append(iou_loss1)
    return sum(losses)

def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (rgbs, irs, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            rgbs, irs, gts = rgbs.cuda(), irs.cuda(), gts.cuda()

            preds, rgb_preds, ir_preds = model(rgbs, irs)

            p1, p2, p3, p4 = preds
            r1, r2, r3, r4 = rgb_preds
            t1, t2, t3, t4 = ir_preds

            loss_rgb = loss_func(r1, gts) + 0.5 * loss_func(r2, gts) + 0.25 * loss_func(r3, gts) + 0.125 * loss_func(r4, gts)
            loss_ir = loss_func(t1, gts) + 0.5 * loss_func(t2, gts) + 0.25 * loss_func(t3, gts) + 0.125 * loss_func(t4, gts)
            loss_aux = 0.5 * (loss_rgb + loss_ir)
            loss_main = loss_func(p1, gts) + 0.5 * loss_func(p2, gts) + 0.25 * loss_func(p3, gts) + 0.125 * loss_func(p4, gts) 
            loss = loss_main + loss_aux

            loss.backward()
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 50 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss_Main: {:.4f} Loss_RGB: {:0.4f} Loss_IR:{:0.4f}'.format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_main.data, loss_rgb.data, loss_ir.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss_Main: {:.4f} Loss_RGB: {:0.4f} Loss_IR:{:0.4f}'.format(epoch, opt.epoch, i, total_step, loss.data, loss_main.data, loss_rgb.data, loss_ir.data))

        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            rgb, ir, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            rgb, ir = rgb.cuda(), ir.cuda()
            pred, _, _ = model(rgb, ir)

            res = F.upsample(pred[0], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
            best_epoch = 1
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=150, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--train_root', type=str, default='/root/autodl-tmp/TVCO1K/train', help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='/root/autodl-tmp/TVCO1K/test', help='the test rgb images root')
    parser.add_argument('--save_path', type=str,default='../snapshot_64_v3_2backbone/TVCO1K/',help='the path to save model and log')
    parser.add_argument('--backbone', type=str, default='resnet101d')
    parser.add_argument('--pretrained_path', type=str, default='../resnet101d.pth')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=1)


    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    cudnn.benchmark = True

    # model = Network(channels=96).cuda()
    model = Network(
        channels=64,
        backbone=opt.backbone,
        pretrained_path=opt.pretrained_path,
    ).cuda()

    if opt.resume is not None:
        print(f"Loading checkpoint from {opt.resume}")
        state_dict = torch.load(opt.resume, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print("Resume model loaded.")


    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(rgb_root=f'{opt.train_root}/RGB/',
                              ir_root=f'{opt.train_root}/T/',
                              gt_root=f'{opt.train_root}/GT/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=4,
                              ir_transform="RGB")
    val_loader = test_dataset(rgb_root=f'{opt.val_root}/RGB/',
                              ir_root=f'{opt.val_root}/T/',
                              gt_root=f'{opt.val_root}/GT/',
                              testsize=opt.trainsize,
                              ir_transform="RGB")
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; save_path: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, save_path))

    step = 0
    best_mae = 1
    best_epoch = 0
    
    # learning rate schedule
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=opt.epoch, eta_min=5e-6)
    print("Start train...")
    for epoch in range(opt.start_epoch, opt.epoch + 1):
        cosine_schedule.step()
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_lr()[0]))
        train(train_loader, model, optimizer, epoch, save_path)
        val(val_loader, model, epoch, save_path)
