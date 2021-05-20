import sys
import argparse
import torch

sys.path.append("/media/a104/D/stli/erfnet/train/tools/")

from model.erfnet import ERFNet
from model.erfnet_pspnet import ERF_PSPNet
from model.erfnet_psp_senet import ERF_PSPSENet
from model.erfnet_sepspnet import ERF_SEPSPNet
from model.erfnet_pspsenet_contactSE import ERF_PSPSE_contactNet
from model.erfnet_sepspnet_contactSE import ERF_SEPSP_contactNet

from ptflops import get_model_complexity_info



pt_models = {

    'ERFNet': ERFNet,
    'ERF_PSPNet': ERF_PSPNet,
    'ERF_PSPSENet': ERF_PSPSENet,
    'ERF_SEPSPNet': ERF_SEPSPNet,
    'ERF_PSPSE_contactNet': ERF_PSPSE_contactNet,
    'ERF_SEPSP_contactNet': ERF_SEPSP_contactNet,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ptflops sample script')
    parser.add_argument('--device', type=int, default=0,
                        help='Device to store the model.')
    parser.add_argument('--model', choices=list(pt_models.keys()),
                        type=str, default='ERF_PSPNet')
    parser.add_argument('--result', type=str, default=None)
    args = parser.parse_args()

    if args.result is None:
        ost = sys.stdout
    else:
        ost = open(args.result, 'w')

    with torch.cuda.device(args.device):
        net = pt_models[args.model](classes=19).cuda()

        flops, params = get_model_complexity_info(net, (3, 512, 1024),
                                                  as_strings=True,
                                                  print_per_layer_stat=True,
                                                  ost=ost)
        print('Flops: ' + flops)
        print('Params: ' + params)
