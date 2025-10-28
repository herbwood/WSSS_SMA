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
import cmapy

def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(tqdm(data_loader)):
            
            # label shape : (20)
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_up_size = imutils.get_strided_up_size(size, 16)

            # len(outputs) : 4
            # outputs[i] shape : (20, h, w)
            # len(pack['img']): 4 
            # img shape : (1, 2, 3, H, W) = (batch size, flipped, channel, height, width)
            # img[0] shape : (2, 3, H, W)
            outputs = [model(img[0].cuda(non_blocking=True))
                       for img in pack['img']] # b x 20 x w x h

            # highres CAM shape : (20, h, w) -> (20, 1, H', W') -> (20, H, W)
            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]

            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]
            valid_cat = torch.nonzero(label)[:, 0] # ex) tensor([0, 1, 3])

            # highres_cam shape : (num labels, H, W)
            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            ###################### CAM visualize ###################### 
            highres_cam, _ = torch.max(highres_cam, 0)
            highres_cam = highres_cam.clone().cpu().detach().numpy() # (H, W)
            img = pack['img'][0][0][0] # (3, H, W)
            image = np.transpose(img.clone().cpu().detach().numpy(), (1,2,0))

            image *= [0.229, 0.224, 0.225]
            image += [0.485, 0.456, 0.406]
            image *= 255
            image = np.clip(image, 0, 255).astype(np.uint8)

            cam_img = highres_cam * 255
            cam_img = np.clip(cam_img, 0, 255)
            cam_img = cv2.applyColorMap(cam_img.astype(np.uint8), cv2.COLORMAP_JET)
            # cam_img = cv2.applyColorMap(cam_img.astype(np.uint8), cmapy.cmap("YlOrRd"))
            image = cv2.addWeighted(image, 0.5, cam_img, 0.5, 0)
            
            cv2.imwrite(os.path.join(args.cam_vis_out_dir, f"{img_name}.png"), image)

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')

def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(torch.load(args.cam_weights_name + '.pth'), strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
                                                             voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()