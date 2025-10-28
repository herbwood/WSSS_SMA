import cv2

import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib

import voc12.dataloader
from misc import pyutils, torchutils
from torch import autograd
import os
import wandb 
from misc.torchutils import SimMinLoss, SimMaxLoss


def validate(model, data_loader, args, ep):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            # labels 
            label = pack['label'].cuda(non_blocking=True)
            bgfg_label = (1 - label).cuda()
            fgbg_label = torch.zeros(label.size()).cuda()
            bgbg_label = torch.ones(label.size()).cuda()
            
            # criterion 
            criterion1 = SimMinLoss()
            criterion2 = SimMaxLoss()
            
            model.zero_grad()

            # losses 
            fgfg, bgfg, fgbg, bgbg, fgbg_concat, fgbg_concat_shuffle, lambda6, z_fg, z_bg, fgbg_concat_shuffle2, lambda8, indices = model(img, ep + 1, args.target_epoch)
            loss_fgfg = F.multilabel_soft_margin_loss(fgfg, label)
            loss_bgfg = F.multilabel_soft_margin_loss(bgfg, bgfg_label)
            loss_fgbg = F.multilabel_soft_margin_loss(fgbg, fgbg_label)
            # loss_bgbg = F.multilabel_soft_margin_loss(bgbg, bgbg_label)
            
            # hie feat label = 0
            loss_bgbg = F.multilabel_soft_margin_loss(bgbg, fgbg_label)
            loss_fgbg_concat = F.multilabel_soft_margin_loss(fgbg_concat, label)
            loss_fgbg_concat_shuffle = F.multilabel_soft_margin_loss(fgbg_concat_shuffle, label)
            loss_cos = criterion1(z_bg, z_fg) + criterion2(z_fg)

            # shuffled label loss 
            shuffled_label = label[indices]
            loss_fgbg_concat_shuffle2 = F.multilabel_soft_margin_loss(fgbg_concat_shuffle2, shuffled_label)

            lambda5 = args.lambda5

            loss = args.lambda1 * loss_fgfg + args.lambda2 * loss_bgfg + args.lambda3 * loss_fgbg \
                    + args.lambda4 * loss_bgbg + lambda5 * loss_fgbg_concat \
                    + lambda6 * loss_fgbg_concat_shuffle \
                    + args.lambda7 * loss_cos \
                    + lambda8 * loss_fgbg_concat_shuffle2
                    
            val_result_dict = {
                          'total_val_loss': loss.item(),
                          'fgfg_val_loss': loss_fgfg.item(),
                          'bgfg_val_loss': loss_bgfg.item(),
                          'fgbg_val_loss': loss_fgbg.item(),
                          'bgbg_val_loss': loss_bgbg.item(),
                          'fgbg_concat_val_loss': loss_fgbg_concat.item(),
                          'fgbg_concat_shuffle_val_loss': loss_fgbg_concat_shuffle.item(),
                          'cos_val_loss': loss_cos.item(),
                          'fgbg_concat_shuffle_val_loss2': loss_fgbg_concat_shuffle2.item(),
                          }

            val_loss_meter.add(val_result_dict)
        
    wandb.log(val_result_dict)

    model.train()

    print(
        f"total val loss:{val_loss_meter.pop('total_val_loss'):.4f}\n",
        f"fgfg val loss:{val_loss_meter.pop('fgfg_val_loss'):.4f}\n",
        f"bgfg val loss:{val_loss_meter.pop('bgfg_val_loss'):.4f}\n",
        f"fgbg val loss:{val_loss_meter.pop('fgbg_val_loss'):.4f}\n",
        f"bgbg val loss:{val_loss_meter.pop('bgbg_val_loss'):.4f}\n",
        f"fgbg concat val loss:{val_loss_meter.pop('fgbg_concat_val_loss'):.4f}\n",
        f"fgbg concat shuffle val loss:{val_loss_meter.pop('fgbg_concat_shuffle_val_loss'):.4f}\n",
        f"cos val loss:{val_loss_meter.pop('cos_val_loss'):.4f}\n",
        f"bgfg concat shuffle val loss:{val_loss_meter.pop('fgbg_concat_shuffle_val_loss2'):.4f}\n",
        )

    return


def run(args):

    model = getattr(importlib.import_module(args.cam_network), 'Net')()


    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    wandb.init(project="DEFT")
    wandb.run.name = args.wandb
    wandb.config.update(args)
    wandb.watch(model)

    for ep in range(args.cam_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.cam_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            img = img.cuda()
            
            # labels 
            label = pack['label'].cuda(non_blocking=True)
            bgfg_label = (1 - label).cuda()
            fgbg_label = torch.zeros(label.size()).cuda()
            bgbg_label = torch.ones(label.size()).cuda()
            
            # criterion 
            criterion1 = SimMinLoss()
            criterion2 = SimMaxLoss()
            
            model.zero_grad()

            # losses 
            fgfg, bgfg, fgbg, bgbg, fgbg_concat, fgbg_concat_shuffle, lambda6, z_fg, z_bg, fgbg_concat_shuffle2, lambda8, indices = model(img, ep + 1, args.target_epoch)
            loss_fgfg = F.multilabel_soft_margin_loss(fgfg, label)
            loss_bgfg = F.multilabel_soft_margin_loss(bgfg, bgfg_label)
            loss_fgbg = F.multilabel_soft_margin_loss(fgbg, fgbg_label)
            # loss_bgbg = F.multilabel_soft_margin_loss(bgbg, bgbg_label)
            
            # hie feat label = 0
            loss_bgbg = F.multilabel_soft_margin_loss(bgbg, fgbg_label)
            loss_fgbg_concat = F.multilabel_soft_margin_loss(fgbg_concat, label)
            loss_fgbg_concat_shuffle = F.multilabel_soft_margin_loss(fgbg_concat_shuffle, label)
            loss_cos = criterion1(z_bg, z_fg) #+ criterion2(z_fg)

            # shuffled label loss 
            shuffled_label = label[indices]
            loss_fgbg_concat_shuffle2 = F.multilabel_soft_margin_loss(fgbg_concat_shuffle2, shuffled_label)

            # lambda5 = args.lambda5
            lambda5 = (ep + 1) / args.cam_num_epoches 
            lambda6 = (ep + 1) / args.cam_num_epoches 
            lambda8 = (ep + 1) / args.cam_num_epoches 

            loss = args.lambda1 * loss_fgfg + args.lambda2 * loss_bgfg + args.lambda3 * loss_fgbg \
                    + args.lambda4 * loss_bgbg + lambda5 * loss_fgbg_concat \
                    + lambda6 * loss_fgbg_concat_shuffle \
                    + args.lambda7 * loss_cos \
                    + lambda8 * loss_fgbg_concat_shuffle2
                    
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            
            train_result_dict = {
                          'total_train_loss': loss.item(),
                          'fgfg_train_loss': loss_fgfg.item(),
                          'bgfg_train_loss': loss_bgfg.item(),
                          'fgbg_train_loss': loss_fgbg.item(),
                          'bgbg_train_loss': loss_bgbg.item(),
                          'fgbg_concat_train_loss': loss_fgbg_concat.item(),
                          'fgbg_concat_shuffle_train_loss': loss_fgbg_concat_shuffle.item(),
                          'cos_train_loss': loss_cos.item(),
                          'fgbg_concat_shuffle_train_loss2': loss_fgbg_concat_shuffle2.item(),
                          }

            avg_meter.add(train_result_dict)


            optimizer.step()
            if (optimizer.global_step-1)%100 == 0:
                timer.update_progress(optimizer.global_step / max_step)
                
                print(f"step{optimizer.global_step - 1}/{max_step}\n",
                      f"total train loss:{avg_meter.pop('total_train_loss'):.4f}\n",
                      f"fgfg train loss:{avg_meter.pop('fgfg_train_loss'):.4f}\n",
                      f"bgfg train loss:{avg_meter.pop('bgfg_train_loss'):.4f}\n",
                      f"fgbg train loss:{avg_meter.pop('fgbg_train_loss'):.4f}\n",
                      f"bgbg train loss:{avg_meter.pop('bgbg_train_loss'):.4f}\n",
                      f"fgbg concat train loss:{avg_meter.pop('fgbg_concat_train_loss'):.4f}\n",
                      f"fgbg concat shuffle train loss:{avg_meter.pop('fgbg_concat_shuffle_train_loss'):.4f}\n",
                      f"cos train loss:{avg_meter.pop('cos_train_loss'):.4f}\n",
                      f"bgfg concat shuffle train loss:{avg_meter.pop('fgbg_concat_shuffle_train_loss2'):.4f}\n",
                      )

        else:
            wandb.log(train_result_dict)
            validate(model, val_data_loader, args, ep)
            timer.reset_stage()

    torch.save(model.module.state_dict(), args.cam_weights_name + '.pth')
    torch.cuda.empty_cache()