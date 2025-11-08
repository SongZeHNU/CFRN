import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from lib.CFRNet import Network
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from shutil import copyfile
from torch.optim import lr_scheduler

def cal_ual(seg_logits, seg_gts):
    seg_logits = seg_logits.contiguous().view(seg_logits.size()[0], -1)
    seg_gts = seg_gts.contiguous().view(seg_gts.size()[0], -1).float()
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    sigmoid_x = seg_logits.sigmoid()

    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
    loss_map = torch.mean(loss_map,1)
    return loss_map.mean()


def dda_loss(pred, mask):

    a = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask) + 1e-6
    b = torch.abs(F.avg_pool2d(mask, kernel_size=51, stride=1, padding=25) - mask) + 1e-6
    c = torch.abs(F.avg_pool2d(mask, kernel_size=61, stride=1, padding=30) - mask) + 1e-6
    d = torch.abs(F.avg_pool2d(mask, kernel_size=27, stride=1, padding=13) - mask) + 1e-6
    e = torch.abs(F.avg_pool2d(mask, kernel_size=21, stride=1, padding=10) - mask) + 1e-6
    alph = 1.75
    
    fall = a**(1.0/(1-alph)) + b**(1.0/(1-alph)) + c**(1.0/(1-alph)) + d**(1.0/(1-alph)) + e**(1.0/(1-alph))
    
    a1 = ((a**(1.0/(1-alph))/fall)**alph)*a
    b1 = ((b**(1.0/(1-alph))/fall)**alph)*b
    c1 = ((c**(1.0/(1-alph))/fall)**alph)*c
    d1 = ((d**(1.0/(1-alph))/fall)**alph)*d
    e1 = ((e**(1.0/(1-alph))/fall)**alph)*e

    weight = 1 + 5* (a1+b1+c1+d1+e1)
    
    dwbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    dwbce = (weight * dwbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weight).sum(dim=(2, 3))
    union = ((pred + mask) * weight).sum(dim=(2, 3))
    dwiou = 1 - (inter + 1) / (union - inter + 1)
    
    return (dwbce + dwiou).mean()  


def train(train_loader, model, optimizer, epoch, save_path, writer):
    #train function
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, edge, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            edge = edge.cuda()
            gts = gts.cuda()
            

            preds = model(images)

            ual = cal_ual(preds[0], gts)
            percentage = epoch / opt.epoch
            coef_range = (0, 1)
            min_coef, max_coef = min(coef_range), max(coef_range)
            normalized_coef = (1 - np.cos(percentage * np.pi)) / 2
            ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
            loss_init = dda_loss(preds[0], gts) 
            loss_final = ual_coef * ual       

            loss =   loss_init + loss_final  

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} '
                    'Loss2: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'Loss_init': loss_init.data, 'Loss_final': loss_final.data,
                                    'Loss_total': loss.data},
                                   global_step=step)
                # TensorboardX-Training Data
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('GT', grid_image, step)



        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer):
    #validation function
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res, _ = model(image)

            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
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
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=3.0e-5, help='learning rate') 
    parser.add_argument('--batch_size', type=int, default=12, help='training batch size') 
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='1', help='train use gpu')
    parser.add_argument('--train_root', type=str, default='./Dataset/TrainValDataset/',    
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='./Dataset/TestDataset/CHAMELEON/',  
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='./snapshot/cfrn_3.0e_100/',
                        help='the path to save model and log')

    #swin argument
    parser.add_argument('--cfg', type=str, default="configs/swin_base_patch4_window12_384.yaml", metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    elif opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')
    elif opt.gpu_id == '4':
        os.environ["CUDA_VISIBLE_DEVICES"] = "4"
        print('USE GPU 4')
    elif opt.gpu_id == '5':
        os.environ["CUDA_VISIBLE_DEVICES"] = "5"
        print('USE GPU 5')
    elif opt.gpu_id == '6':
        os.environ["CUDA_VISIBLE_DEVICES"] = "6"
        print('USE GPU 6')
    elif opt.gpu_id == '7':
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        print('USE GPU 7')
    cudnn.benchmark = True

    # build the model
    model = Network(opt).cuda()

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #record every run
    copyfile('./MyTrain_Val.py', save_path+'/MyTrain_Val.py')
    copyfile('./lib/CFRNet.py', save_path+'/CFRNet.py')

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/',
                              edge_root=opt.train_root + 'Edge/',
                              gt_root=opt.train_root + 'GT/',
                              batchsize=opt.batch_size,
                              trainsize=opt.trainsize,
                              num_workers=4)
    val_loader = test_dataset(image_root=opt.val_root + 'Imgs/',
                              gt_root=opt.val_root + 'GT/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batch_size, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_epoch = 0

    print("Start train...")
    for epoch in range(1, opt.epoch+1):
        train(train_loader, model, optimizer, epoch, save_path, writer)
        exp_lr_scheduler.step()
        val(val_loader, model, epoch, save_path, writer)
