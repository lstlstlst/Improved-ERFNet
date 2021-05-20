"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import time
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

##import test_config
import torch
from datasets import get_dataset
from models import get_model
from utils.utils import Cluster, Visualizer

torch.backends.cudnn.benchmark = True

####################################################################
from PIL import Image
from utils import transforms as my_transforms
from transform import Colorize

#os.environ.get('CITYSCAPES_DIR')
CITYSCAPES_DIR='/media/a104/D/stli/data/cityscapes/'


args = dict(

    cuda=True,
    display=False,

    save=True,
    save_dir='/media/a104/D/stli/erfnet/save/instance_training/infer_label/',
    checkpoint_path='/media/a104/D/stli/erfnet/save/instance_training/checkpoint.pth',

    dataset= { 
        'name': 'cityscapes',
        'kwargs': {
            'root_dir': CITYSCAPES_DIR,
            'type': 'val',
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        }
    },
        
    model = {
        'name': 'branched_erfnet',
        'kwargs': {
            'num_classes': [3, 1],
        }
    }
)

######################################################################
if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# dataloader
dataset = get_dataset(
    args['dataset']['name'], args['dataset']['kwargs'])
dataset_it = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True if args['cuda'] else False)

# load model
model = get_model(args['model']['name'], args['model']['kwargs'])
model = torch.nn.DataParallel(model).to(device)

# load snapshot
if os.path.exists(args['checkpoint_path']):
    state = torch.load(args['checkpoint_path'])
    model.load_state_dict(state['model_state_dict'], strict=True)
else:
    assert(False, 'checkpoint_path {} does not exist!'.format(args['checkpoint_path']))

model.eval()

# cluster module
cluster = Cluster()

# Visualizer
visualizer = Visualizer(('image', 'pred', 'sigma', 'seed'))

with torch.no_grad():

    for sample in tqdm(dataset_it):

        im = sample['image']
        instances = sample['instance'].squeeze()
        
        output = model(im)
        instance_map, predictions = cluster.cluster(output[0], threshold=0.9)


        visualizer.display(im, 'image')

            
        visualizer.display(instance_map.cpu(), 'pred')
        label1 = instance_map.cpu()
        label1.squeeze_()

        sigma = output[0][2].cpu()
        sigma = (sigma - sigma.min())/(sigma.max() - sigma.min())
        sigma[instances == 0] = 0
        visualizer.display(sigma, 'sigma')
        label2 = sigma
        label2.squeeze_()

        seed = torch.sigmoid(output[0][3]).cpu()
        visualizer.display(seed, 'seed')
        label3 = seed.cpu()
        label3.squeeze_()



        if args['save']:
            
            base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
            
            txt_file = os.path.join(args['save_dir'], base + '.txt')
            
            im_name = base + '.png'

            #im = torchvision.transforms.ToPILImage()(pred['mask'].unsqueeze(0))
            im = torchvision.transforms.ToPILImage()(label2)

            # write image
            im.save(os.path.join(args['save_dir'], im_name))


