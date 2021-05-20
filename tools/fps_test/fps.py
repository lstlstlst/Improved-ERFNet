import torch
import oyaml as yaml
from torchstat import stat
import time,os

import sys
sys.path.append("/media/a104/D/stli/erfnet/train/tools/")
from model.erfnet import ERFNet
from model.erfnet_pspnet import ERF_PSPNet
from model.erfnet_psp_senet import ERF_PSPSENet
from model.erfnet_sepspnet import ERF_SEPSPNet
from model.erfnet_pspsenet_contactSE import ERF_PSPSE_contactNet
from model.erfnet_sepspnet_contactSE import ERF_SEPSP_contactNet

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def run(model,size,name):
    model.cuda()
    model.eval()
    t_cnt = 0.0
    with torch.no_grad():

        input = torch.rand(size).cuda()
        torch.cuda.synchronize()
        x = model(input)
        x = model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_ts = time.time()

        for i in range(1000):
            x = model(input)
        torch.cuda.synchronize()
        end_ts = time.time()

        t_cnt = end_ts-start_ts #t_cnt + (end_ts-start_ts)
        speed_time = t_cnt / 1000 * 1000#后面1000是毫秒

    print("=======================================")
    print("Model Name: "+name)
    print('Elapsed Time: [%.2f s / %d iter]' % (t_cnt, 1000))
    print("Speed Time: %.2f ms / iter  FPS: %f"%(speed_time, 1000/t_cnt))
    #print("=======================================")



if __name__ == "__main__":

    erfnet = ERFNet(classes =19)
    run(erfnet,size=(1,3,512,1024),name='ERFNet')

    erfnet_pspnet = ERF_PSPNet(classes =19)
    run(erfnet_pspnet,size=(1,3,512,1024),name='ERF_PSPNet')

    erfnet_sepspnet = ERF_SEPSPNet(classes =19)
    run(erfnet_sepspnet,size=(1,3,512,1024),name='ERF_SEPSPNet')

    erfnet_pspsenet = ERF_PSPSENet(classes =19)
    run(erfnet_pspsenet,size=(1,3,512,1024),name='ERF_PSPSENet')

    erfnet_sepspnet_contactSE = ERF_SEPSP_contactNet(classes =19)
    run(erfnet_sepspnet,size=(1,3,512,1024),name='ERF_SEPSP_contactNet')

    erfnet_pspsenet_contactSE = ERF_PSPSE_contactNet(classes =19)
    run(erfnet_sepspnet,size=(1,3,512,1024),name='ERF_PSPSE_contactNet')
