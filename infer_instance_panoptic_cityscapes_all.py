import os
import torch
import torch.nn as nn
import argparse
import cv2
import numpy as np
import sys
sys.path.append("/media/a104/D/stli/UPSNet-master/")

from upsnet.config.config import *
from upsnet.config.parse_args import parse_args

from upsnet.models import *

from PIL import Image, ImageDraw


def get_pallete():


 pallete_raw = np.zeros((256, 3)).astype('uint8')
 pallete = np.zeros((256, 3)).astype('uint8')

 pallete_raw[5, :] = [111, 74, 0]
 pallete_raw[6, :] = [81, 0, 81]
 pallete_raw[7, :] = [128, 64, 128]
 pallete_raw[8, :] = [244, 35, 232]
 pallete_raw[9, :] = [250, 170, 160]
 pallete_raw[10, :] = [230, 150, 140]
 pallete_raw[11, :] = [70, 70, 70]
 pallete_raw[12, :] = [102, 102, 156]
 pallete_raw[13, :] = [190, 153, 153]
 pallete_raw[14, :] = [180, 165, 180]
 pallete_raw[15, :] = [150, 100, 100]
 pallete_raw[16, :] = [150, 120, 90]
 pallete_raw[17, :] = [153, 153, 153]
 pallete_raw[18, :] = [153, 153, 153]
 pallete_raw[19, :] = [250, 170, 30]
 pallete_raw[20, :] = [220, 220, 0]
 pallete_raw[21, :] = [107, 142, 35]
 pallete_raw[22, :] = [152, 251, 152]
 pallete_raw[23, :] = [70, 130, 180]
 pallete_raw[24, :] = [220, 20, 60]
 pallete_raw[25, :] = [255, 0, 0]
 pallete_raw[26, :] = [0, 0, 142]
 pallete_raw[27, :] = [0, 0, 70]
 pallete_raw[28, :] = [0, 60, 100]
 pallete_raw[29, :] = [0, 0, 90]
 pallete_raw[30, :] = [0, 0, 110]
 pallete_raw[31, :] = [0, 80, 100]
 pallete_raw[32, :] = [0, 0, 230]
 pallete_raw[33, :] = [119, 11, 32]

 train2regular = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

 for i in range(len(train2regular)):
    pallete[i, :] = pallete_raw[train2regular[i], :]

 pallete = pallete.reshape(-1)

# return pallete_raw
 return pallete


parser = argparse.ArgumentParser()
args, rest = parser.parse_known_args()
args.cfg = "/media/a104/D/stli/erfnet/train/cfg/panoptic_cityscapes.yaml"
#第一个创新点
args.weight_path = "/media/a104/D/stli/erfnet/save/instance_training/checkpoint_cityscapes.pth"
#第二个创新点
#args.weight_path = "/media/a104/D/stli/erfnet/save/panoptic_training/checkpoint_cityscapes.pth"

args.eval_only = "Ture"
update_config(args.cfg)

test_model = eval("resnet_50_upsnet")().cuda()
test_model.load_state_dict(torch.load(args.weight_path))

#print(test_model)

for p in test_model.parameters():
    p.requires_grad = False

test_model.eval()

dat = []
for line in open("/media/a104/D/stli/erfnet/train/cfg/train_cityscapes.txt","r"):
    dat.append(line.strip('\n'))

for i in range(len(dat)):
    im = cv2.imread("/media/a104/D/stli/erfnet/train/data/cityscapes/images/"+dat[i])
    im_resize = cv2.resize(im, (2048, 1024), interpolation=cv2.INTER_CUBIC)
    im_resize = im_resize.transpose(2, 0, 1)
    
    im_tensor = torch.from_numpy(im_resize)
    im_tensor = torch.unsqueeze(im_tensor, 0).type(torch.FloatTensor).cuda()
    print(im_tensor.shape)

    
    #全景分割
    im_info = np.array([[1024, 2048, 3]])
    data = {'data': im_tensor , 'im_info' : im_info}
    
    print(data['im_info'])
    output = test_model(data)
    
    print(output['panoptic_outputs'])
    
    pallete = get_pallete()
    output=output['panoptic_outputs'].cuda().data.cpu().numpy()
    segmentation_result = np.copy(output)
    segmentation_result = np.uint8(np.squeeze(segmentation_result))
    segmentation_result = Image.fromarray(segmentation_result)
    segmentation_result.putpalette(pallete)
    segmentation_result = segmentation_result.resize((im.shape[1], im.shape[0]))
#第一个创新点 
    segmentation_result.save("/media/a104/D/stli/erfnet/save/instance_training/infer/"+dat[i])
#第二个创新点
   # segmentation_result.save("/media/a104/D/stli/erfnet/save/panoptic_training/infer_cityscapes/"+dat[i])
