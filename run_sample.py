import argparse
import os
import numpy as np

from misc import pyutils

if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--voc12_root", type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")

    # Dataset
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
    parser.add_argument("--chainer_eval_set", default="train", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int) # original: 16
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.15, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.30, type=float)
    parser.add_argument("--conf_bg_thres", default=0.05, type=float)

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="net.resnet50_irn", type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=32, type=int)
    parser.add_argument("--irn_num_epoches", default=3, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--ins_seg_bg_thres", default=0.25)
    parser.add_argument("--sem_seg_bg_thres", default=0.25)

    # Output Path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--cam_weights_name", default="sess/res50_cam.pth", type=str)
    parser.add_argument("--irn_weights_name", default="sess/res50_irn.pth", type=str)
    parser.add_argument("--cam_out_dir", default="result/cam_adv_mask", type=str)
    parser.add_argument("--ir_label_out_dir", default="result/ir_label", type=str)
    parser.add_argument("--sem_seg_out_dir", default="result/sem_seg", type=str)
    parser.add_argument("--ins_seg_out_dir", default="result/ins_seg", type=str)
    parser.add_argument("--cam_vis_out_dir", default="result/cam_vis", type=str)

    # Step
    parser.add_argument("--train_cam_pass", action="store_true")
    parser.add_argument("--make_cam_pass", action="store_true")
    parser.add_argument("--eval_cam_pass", action="store_true")
    parser.add_argument("--cam_to_ir_label_pass", action="store_true")
    parser.add_argument("--train_irn_pass", action="store_true")
    parser.add_argument("--make_sem_seg_pass", action="store_true")
    parser.add_argument("--eval_sem_seg_pass", action="store_true")

    # DEFT argparsers 
    
    # steps 
    parser.add_argument("--train_cam_sma_pass", action="store_true")
    parser.add_argument("--make_cam_sma_pass", action="store_true")
    parser.add_argument("--cam_vis_pass", action="store_true")
    parser.add_argument("--cam_vis_sma_pass", action="store_true")

    # wandb and descriptions 
    parser.add_argument("--wandb", type=str)
    parser.add_argument("--wandb_project", type=str, default="SMA")
    parser.add_argument("--description", type=str)
    
    # hyperparameters 
    parser.add_argument("--target_epoch", type=int)
    
    parser.add_argument("--lambda1", type=int,
                            default=1, help="fg2fg loss balancing scalar")
    parser.add_argument("--lambda2", type=float,
                        default=0.3, help="fg2bg loss balancing scalar")
    parser.add_argument("--lambda3", type=float,
                        default=0.3, help="bg2fg loss balancing scalar")
    parser.add_argument("--lambda4", type=float,
                        default=0.2, help="bg2bg loss balancing scalar")
    parser.add_argument("--lambda5", type=float,
                        default=1, help="concat loss balancing scalar")
    parser.add_argument("--lambda6", type=float,
                        default=1, help="sim min cos loss balancing scalar")
    parser.add_argument("--lambda7", type=float,
                        default=1, help="sim max fg loss balancing scalar")
    parser.add_argument("--lambda8", type=float,
                        default=0, help="sim max bg loss balancing scalar")
    parser.add_argument("--cam_power", default=1.0, type=float)
    parser.add_argument("--gamma", default=0.5, type=float)

    args = parser.parse_args()

    os.makedirs(f"wsss_sma/{args.wandb}/sess", exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.ir_label_out_dir, exist_ok=True)
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)
    os.makedirs(args.ins_seg_out_dir, exist_ok=True)
    os.makedirs(args.cam_vis_out_dir, exist_ok=True)
    pyutils.Logger(args.log_name + '.log')
    print(vars(args))

    if args.train_cam_pass is True:
        import step.train_cam

        timer = pyutils.Timer('step.train_cam:')
        step.train_cam.run(args)
    
    # on test 
    if args.train_cam_sma_pass is True:
        import step.train_cam_sma

        timer = pyutils.Timer('step.train_cam_sma:')
        step.train_cam_sma.run(args)
    
    if args.make_cam_sma_pass is True:
        import step.make_cam_sma

        timer = pyutils.Timer('step.make_cam_sma:')
        step.make_cam_sma.run(args)
        
    if args.cam_vis_pass is True:
        import misc.cam_vis 

        timer = pyutils.Timer('misc.cam_vis:')
        misc.cam_vis.run(args)

    if args.eval_cam_pass is True:
        import step.eval_cam

        timer = pyutils.Timer('step.eval_cam:')
        step.eval_cam.run(args)

    if args.cam_to_ir_label_pass is True:
        import step.cam_to_ir_label

        timer = pyutils.Timer('step.cam_to_ir_label:')
        step.cam_to_ir_label.run(args)

    if args.train_irn_pass is True:
        import step.train_irn

        timer = pyutils.Timer('step.train_irn:')
        step.train_irn.run(args)

    if args.make_sem_seg_pass is True:
        import step.make_sem_seg_labels
        
        timer = pyutils.Timer('step.make_sem_seg_labels:')
        step.make_sem_seg_labels.run(args)

    if args.eval_sem_seg_pass is True:
        import step.eval_sem_seg

        timer = pyutils.Timer('step.eval_sem_seg:')
        step.eval_sem_seg.run(args)