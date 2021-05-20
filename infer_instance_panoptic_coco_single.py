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

 pallete_raw[1, :] = [220, 20, 60]
 pallete_raw[2, :] = [119, 11, 32]
 pallete_raw[3, :] = [0, 0, 142]
 pallete_raw[4, :] = [0, 0, 230]
 pallete_raw[5, :] = [106, 0, 228]
 pallete_raw[6, :] = [0, 60, 100]
 pallete_raw[7, :] = [0, 80, 100]
 pallete_raw[8, :] = [0, 0, 70]
 pallete_raw[9, :] = [0, 0, 192]
 pallete_raw[10, :] = [250, 170, 30]
 pallete_raw[11, :] = [100, 170, 30]
 pallete_raw[13, :] = [220, 220, 0]
 pallete_raw[14, :] = [175, 116, 175]
 pallete_raw[15, :] = [250, 0, 30]
 pallete_raw[16, :] = [165, 42, 42]
 pallete_raw[17, :] = [255, 77, 255]
 pallete_raw[18, :] = [0, 226, 252]
 pallete_raw[19, :] = [182, 182, 255]
 pallete_raw[20, :] = [0, 82, 0]
 pallete_raw[21, :] = [120, 166, 157]
 pallete_raw[22, :] = [110, 76, 0]
 pallete_raw[23, :] = [174, 57, 255]
 pallete_raw[24, :] = [199, 100, 0]
 pallete_raw[25, :] = [72, 0, 118]
 pallete_raw[27, :] = [255, 179, 240]
 pallete_raw[28, :] = [0, 125, 92]
 pallete_raw[31, :] = [209, 0, 151]
 pallete_raw[32, :] = [188, 208, 182]
 pallete_raw[33, :] = [0, 220, 176]
 pallete_raw[34, :] = [255, 99, 164]
 pallete_raw[35, :] = [92, 0, 73]
 pallete_raw[36, :] = [133, 129, 255]
 pallete_raw[37, :] = [78, 180, 255]
 pallete_raw[38, :] = [0, 228, 0]
 pallete_raw[39, :] = [174, 255, 243]
 pallete_raw[40, :] = [45, 89, 255]
 pallete_raw[41, :] = [134, 134, 103]
 pallete_raw[42, :] = [145, 148, 174]
 pallete_raw[43, :] = [255, 208, 186]
 pallete_raw[44, :] = [197, 226, 255]
 pallete_raw[46, :] = [171, 134, 1]
 pallete_raw[47, :] = [109, 63, 54]
 pallete_raw[48, :] = [207, 138, 255]
 pallete_raw[49, :] = [151, 0, 95]
 pallete_raw[50, :] = [9, 80, 61]
 pallete_raw[51, :] = [84, 105, 51]
 pallete_raw[52, :] = [74, 65, 105]
 pallete_raw[53, :] = [166, 196, 102]
 pallete_raw[54, :] = [208, 195, 210]
 pallete_raw[55, :] = [255, 109, 65]
 pallete_raw[56, :] = [0, 143, 149]
 pallete_raw[57, :] = [179, 0, 194]
 pallete_raw[58, :] = [209, 99, 106]
 pallete_raw[59, :] = [5, 121, 0]
 pallete_raw[60, :] = [227, 255, 205]
 pallete_raw[61, :] = [147, 186, 208]
 pallete_raw[62, :] = [153, 69, 1]
 pallete_raw[63, :] = [3, 95, 161]
 pallete_raw[64, :] = [163, 255, 0]
 pallete_raw[65, :] = [119, 0, 170]
 pallete_raw[67, :] = [0, 182, 199]
 pallete_raw[70, :] = [0, 165, 120]
 pallete_raw[72, :] = [183, 130, 88]
 pallete_raw[73, :] = [95, 32, 0]
 pallete_raw[74, :] = [130, 114, 135]
 pallete_raw[75, :] = [110, 129, 133]
 pallete_raw[76, :] = [166, 74, 118]
 pallete_raw[77, :] = [219, 142, 185]
 pallete_raw[78, :] = [79, 210, 114]
 pallete_raw[79, :] = [178, 90, 62]
 pallete_raw[80, :] = [65, 70, 15]
 pallete_raw[81, :] = [127, 167, 115]
 pallete_raw[82, :] = [59, 105, 106]
 pallete_raw[84, :] = [142, 108, 45]
 pallete_raw[85, :] = [196, 172, 0]
 pallete_raw[86, :] = [95, 54, 80]
 pallete_raw[87, :] = [128, 76, 255]
 pallete_raw[88, :] = [201, 57, 1]
 pallete_raw[89, :] = [246, 0, 122]
 pallete_raw[90, :] = [191, 162, 208]
 pallete_raw[92, :] = [255, 255, 128]
 pallete_raw[93, :] = [147, 211, 203]
 pallete_raw[95, :] = [150, 100, 100]
 pallete_raw[100, :] = [168, 171, 172]
 pallete_raw[107, :] = [146, 112, 198]
 pallete_raw[109, :] = [210, 170, 100]
 pallete_raw[112, :] = [92, 136, 89]
 pallete_raw[118, :] = [218, 88, 184]
 pallete_raw[119, :] = [241, 129, 0]
 pallete_raw[122, :] = [217, 17, 255]
 pallete_raw[125, :] = [124, 74, 181]
 pallete_raw[128, :] = [70, 70, 70]
 pallete_raw[130, :] = [255, 228, 255]
 pallete_raw[133, :] = [154, 208, 0]
 pallete_raw[138, :] = [193, 0, 92]
 pallete_raw[141, :] = [76, 91, 113]
 pallete_raw[144, :] = [255, 180, 195]
 pallete_raw[145, :] = [106, 154, 176]
 pallete_raw[147, :] = [230, 150, 140]
 pallete_raw[148, :] = [60, 143, 255]
 pallete_raw[149, :] = [128, 64, 128]
 pallete_raw[151, :] = [92, 82, 55]
 pallete_raw[154, :] = [254, 212, 124]
 pallete_raw[155, :] = [73, 77, 174]
 pallete_raw[156, :] = [255, 160, 98]
 pallete_raw[159, :] = [255, 255, 255]
 pallete_raw[161, :] = [104, 84, 109]
 pallete_raw[166, :] = [169, 164, 131]
 pallete_raw[168, :] = [225, 199, 255]
 pallete_raw[171, :] = [137, 54, 74]
 pallete_raw[175, :] = [135, 158, 223]
 pallete_raw[176, :] = [7, 246, 231]
 pallete_raw[177, :] = [107, 255, 200]
 pallete_raw[178, :] = [58, 41, 149]
 pallete_raw[180, :] = [183, 121, 142]
 pallete_raw[181, :] = [255, 73, 97]
 pallete_raw[184, :] = [107, 142, 35]
 pallete_raw[185, :] = [190, 153, 153]
 pallete_raw[186, :] = [146, 139, 141]
 pallete_raw[187, :] = [70, 130, 180]
 pallete_raw[188, :] = [134, 199, 156]
 pallete_raw[189, :] = [209, 226, 140]
 pallete_raw[190, :] = [96, 36, 108]
 pallete_raw[191, :] = [96, 96, 96]
 pallete_raw[192, :] = [64, 170, 64]
 pallete_raw[193, :] = [152, 251, 152]
 pallete_raw[194, :] = [208, 229, 228]
 pallete_raw[195, :] = [206, 186, 171]
 pallete_raw[196, :] = [152, 161, 64]
 pallete_raw[197, :] = [116, 112, 0]
 pallete_raw[198, :] = [0, 114, 143]
 pallete_raw[199, :] = [102, 102, 156]
 pallete_raw[200, :] = [250, 141, 255]


 #train2regular = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
 train2regular = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 92, 93, 95, 100, 107, 109, 112, 118, 119, 122, 125, 128, 130, 133, 138, 141, 144, 145, 147, 148, 149, 151, 154, 155, 156, 159, 161, 166, 168, 171, 175, 176, 177, 178, 180, 181, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
 for i in range(len(train2regular)):
    pallete[i, :] = pallete_raw[train2regular[i], :]

 pallete = pallete.reshape(-1)

# return pallete_raw
 return pallete


parser = argparse.ArgumentParser()
args, rest = parser.parse_known_args()
args.cfg = "/media/a104/D/stli/erfnet/train/cfg/panoptic_coco.yaml"
#第一个创新点
args.weight_path = "/media/a104/D/stli/erfnet/save/instance_training/checkpoint_coco.pth"
#第二个创新点
#args.weight_path = "/media/a104/D/stli/erfnet/save/panoptic_training/checkpoint_coco.pth"

args.eval_only = "Ture"
update_config(args.cfg)

test_model = eval("resnet_50_upsnet")().cuda()
test_model.load_state_dict(torch.load(args.weight_path))

#print(test_model)

for p in test_model.parameters():
    p.requires_grad = False

test_model.eval()

im = cv2.imread("/media/a104/D/stli/erfnet/train/data/coco/images/val2017/000000000632.jpg")
im_resize = cv2.resize(im, (2048, 1024), interpolation=cv2.INTER_CUBIC)
im_resize = im_resize.transpose(2, 0, 1)

im_tensor = torch.from_numpy(im_resize)
im_tensor = torch.unsqueeze(im_tensor, 0).type(torch.FloatTensor).cuda()
print(im_tensor.shape)  # torch.Size([1, 3, 1024, 2048])

#实例分割：修改fcn_outputs为panoptic_outputs
test_fake_numpy_data = np.random.rand(1, 3)
data = {'data': im_tensor, 'im_info': test_fake_numpy_data}

#全景分割：修改fcn_outputs为panoptic_outputs
#im_info = np.array([[1024, 2048, 3]])
#data = {'data': im_tensor , 'im_info' : im_info}

print(data['im_info'])
output = test_model(data)
#print(output)
#print(output['panoptic_outputs'])

pallete = get_pallete()
output=output['panoptic_outputs'].cuda().data.cpu().numpy()
segmentation_result = np.copy(output)
segmentation_result = np.uint8(np.squeeze(segmentation_result))
segmentation_result = Image.fromarray(segmentation_result)
segmentation_result.putpalette(pallete)
segmentation_result = segmentation_result.resize((im.shape[1], im.shape[0]))
#第一个创新点
segmentation_result.save("/media/a104/D/stli/erfnet/train/000000000632_1.png")
#第二个创新点
#segmentation_result.save("/media/a104/D/stli/erfnet/train/000000000632_2.png")

