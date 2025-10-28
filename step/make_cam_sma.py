import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
from tqdm import tqdm 

import voc12.dataloader
from misc import torchutils, imutils
import cv2
cudnn.enabled = True

def _work(model, dataset, args):

    # databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(dataset, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad():

        model.cuda()

        for iter, pack in enumerate(tqdm(data_loader)):
            
            # label shape : (20)
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            # len(outputs) : 4
            # outputs[i] shape : (20, h, w)
            # len(pack['img']): 4 
            # img shape : (1, 2, 3, H, W) = (batch size, flipped, channel, height, width)
            # img[0] shape : (2, 3, H, W)
            outputs = [model(img[0].cuda(non_blocking=True))
                       for img in pack['img']] # b x 20 x w x h

            # CAM shape : (20, h, w) -> (1, 20, h, w) -> (1, 20, h', w') -> (4, 20, h', w') -> (20, h', w')
            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs]), 0)

            # highres CAM shape : (20, h, w) -> (20, 1, H', W') -> (20, H, W)
            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]

            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
            valid_cat = torch.nonzero(label)[:, 0] # ex) tensor([0, 1, 3])

            # strided_cam shape : (num labels, h', w')
            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            # highres_cam shape : (num labels, H, W)
            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            # cv2.imshow('highres', (highres_cam[0].cpu().numpy()*255.0).astype('uint8'))
            # cv2.waitKey(0)
            # save cams
            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model.eval()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
                                                             voc12_root=args.voc12_root, scales=args.cam_scales)

    print('[ ', end='')
    _work(model, dataset, args)
    print(']')

    torch.cuda.empty_cache()