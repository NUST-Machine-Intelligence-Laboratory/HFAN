import os
import os.path as osp
import cv2
from glob import glob
import numpy as np
import mmcv

import time
import argparse
import os
from os import system

import torch
from mmcv.utils import DictAction
from tools.test import main


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('--config', help='test config file path',
                        default='local_configs/hfan/hfan.small.512x512.refine.py'
                        )
    parser.add_argument('--checkpoint', help='checkpoint file',
                        default='checkpoint/HFAN-s-converted.pth'
                        )
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument(
        '--fast-test',
        type=str,
        default='sp',
        help='Parallel: sp or mp')
    
    parser.add_argument("--input_dir", default='/DAVIS/DAVIS2SEG/', help="input path", type=str)

    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
             ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', "--output_dir", help='directory where painted images will be saved',
        default='./output_path/hfan'
    )
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options', default={
            'data.test.img1_dir': 'frame/val',
            'data.test.img2_dir': 'flow/val',
            'data.workers_per_gpu': 4,
        })
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=1,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def archive(pred_path, out_dir):
    pred_list = os.listdir(pred_path)
    for pred_name in pred_list:
        im_name = pred_name[-9:-4]
        video_name = pred_name[:-10]
        pred = mmcv.imread(osp.join(pred_path, pred_name))
        pred = pred[:, :, 2]
        pred[pred >= 127] = 255
        pred[pred < 127] = 0
        out_path = osp.join(out_dir, video_name, im_name + '.png')
        mmcv.imwrite(pred, out_path)


if __name__ == '__main__':
    # arguments and environments setting
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    start_time = time.time()
    torch.backends.cudnn.benchmark = True

    # inference fused results
    args = parse_args()
    args.options.update({'data.test.data_root': args.input_dir})
    main(args)
    archive(args.show_dir, args.show_dir + '-DAVIS-16')
    print('\n total time:', time.time() - start_time)

