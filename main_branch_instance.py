
import os
import shutil
import time

from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
#import train_config
from criterions.my_loss import SpatialEmbLoss
from datasets import get_dataset
from models import get_model
from utils.utils import AverageMeter, Cluster, Logger, Visualizer

torch.backends.cudnn.benchmark = True


from PIL import Image
from utils import transforms as my_transforms

#os.environ.get('CITYSCAPES_DIR')
CITYSCAPES_DIR='/media/a104/D/stli/data/cityscapes/'

#################初始参数设置###########################
args = dict(

    cuda=True,
    display=False,
    display_it=5,

    save=True,
    save_dir='/media/a104/D/stli/erfnet/save/instance_training',
    resume_path=None, 

    train_dataset = {
        'name': 'cityscapes',
        'kwargs': {
            'root_dir': CITYSCAPES_DIR,
            'type': 'crops',
            'size': 3000,
            'transform': my_transforms.get_transform([
                {
                    'name': 'RandomCrop',
                    'opts': {
                        'keys': ('image', 'instance','label'),
                        'size': (512, 512),
                    }
                },
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        'batch_size': 16,
        'workers': 0
    }, 

    val_dataset = {
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
        },
        'batch_size': 16,
        'workers': 0
    }, 

    model = {
        'name': 'branched_erfnet', 
        'kwargs': {
            'num_classes': [3,1]
        }
    }, 

    lr=5e-4,
    n_epochs=200,

    # loss options
    loss_opts={
        'to_center': True,
        'n_sigma': 1,
        'foreground_weight': 10,
    },
    loss_w={
        'w_inst': 1,
        'w_var': 10,
        'w_seed': 1,
    },
)
#######################################################################################
if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])

if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

# 设置cpu,gpu
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# 数据集设置
train_dataset = get_dataset(
    args['train_dataset']['name'], args['train_dataset']['kwargs'])
train_dataset_it = torch.utils.data.DataLoader(
    train_dataset, batch_size=args['train_dataset']['batch_size'], shuffle=True, drop_last=True, num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

val_dataset = get_dataset(
    args['val_dataset']['name'], args['val_dataset']['kwargs'])
val_dataset_it = torch.utils.data.DataLoader(
    val_dataset, batch_size=args['val_dataset']['batch_size'], shuffle=True, drop_last=True, num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)


# 模型调用
model = get_model(args['model']['name'], args['model']['kwargs'])
model.init_output(args['loss_opts']['n_sigma'])
model = torch.nn.DataParallel(model).to(device)

# loss函数调用
criterion = SpatialEmbLoss(**args['loss_opts'])
criterion = torch.nn.DataParallel(criterion).to(device)

# 优化
optimizer = torch.optim.Adam(
    model.parameters(), lr=args['lr'], weight_decay=1e-4)

def lambda_(epoch):
    return pow((1-((epoch)/args['n_epochs'])), 0.9)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda_,)

# 聚类
cluster = Cluster()

# 可视化
visualizer = Visualizer(('image', 'pred', 'sigma', 'seed'))

# 融合
logger = Logger(('train', 'val', 'iou'), 'loss')

# epoch从0开始
start_epoch = 0
best_iou = 0
if args['resume_path'] is not None and os.path.exists(args['resume_path']):
    print('Resuming model from {}'.format(args['resume_path']))
    state = torch.load(args['resume_path'])
    start_epoch = state['epoch'] + 1
    best_iou = state['best_iou']
    model.load_state_dict(state['model_state_dict'], strict=True)
    optimizer.load_state_dict(state['optim_state_dict'])
    logger.data = state['logger_data']

# 训练
def train(epoch):

    # define meters
    loss_meter = AverageMeter()

    # put model into training mode
    model.train()

    for param_group in optimizer.param_groups:
        print('learning rate: {}'.format(param_group['lr']))

    for i, sample in enumerate(tqdm(train_dataset_it)):

        im = sample['image']
        instances = sample['instance'].squeeze()
        class_labels = sample['label'].squeeze()

        output = model(im)
        loss = criterion(output, instances, class_labels, **args['loss_w'])
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args['display'] and i % args['display_it'] == 0:
            with torch.no_grad():
                visualizer.display(im[0], 'image')
                
                predictions = cluster.cluster_with_gt(output[0], instances[0], n_sigma=args['loss_opts']['n_sigma'])
                visualizer.display([predictions.cpu(), instances[0].cpu()], 'pred')

                sigma = output[0][2].cpu()
                sigma = (sigma - sigma.min())/(sigma.max() - sigma.min())
                sigma[instances[0] == 0] = 0
                visualizer.display(sigma, 'sigma')

                seed = torch.sigmoid(output[0][3]).cpu()
                visualizer.display(seed, 'seed')

        loss_meter.update(loss.item())

    return loss_meter.avg

# 验证
def val(epoch):

    
    loss_meter, iou_meter = AverageMeter(), AverageMeter()

    
    model.eval()

    with torch.no_grad():

        for i, sample in enumerate(tqdm(val_dataset_it)):

            im = sample['image']
            instances = sample['instance'].squeeze()
            class_labels = sample['label'].squeeze()

            output = model(im)
            loss = criterion(output, instances, class_labels, **
                            args['loss_w'], iou=True, iou_meter=iou_meter)
            loss = loss.mean()

            if args['display'] and i % args['display_it'] == 0:
                with torch.no_grad():
                    visualizer.display(im[0], 'image')
                
                    predictions = cluster.cluster_with_gt(output[0], instances[0], n_sigma=args['loss_opts']['n_sigma'])
                    visualizer.display([predictions.cpu(), instances[0].cpu()], 'pred')
    
                    sigma = output[0][2].cpu()
                    sigma = (sigma - sigma.min())/(sigma.max() - sigma.min())
                    sigma[instances[0] == 0] = 0
                    visualizer.display(sigma, 'sigma')
    
                    seed = torch.sigmoid(output[0][3]).cpu()
                    visualizer.display(seed, 'seed')

            loss_meter.update(loss.item())

    return loss_meter.avg, iou_meter.avg

# 保存模型
def save_checkpoint(state, is_best, name='checkpoint.pth'):
    print('=> saving checkpoint')
    file_name = os.path.join(args['save_dir'], name)
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_iou_model.pth'))

for epoch in range(start_epoch, args['n_epochs']):

    print('Starting epoch {}'.format(epoch))
    scheduler.step(epoch)

    train_loss = train(epoch)
    val_loss, val_iou = val(epoch)

    print('===> train loss: {:.2f}'.format(train_loss))
    print('===> val loss: {:.2f}, val iou: {:.2f}'.format(val_loss, val_iou))

    logger.add('train', train_loss)
    logger.add('val', val_loss)
    logger.add('iou', val_iou)
    logger.plot(save=args['save'], save_dir=args['save_dir'])
    
    is_best = val_iou > best_iou
    best_iou = max(val_iou, best_iou)
        
    if args['save']:
        state = {
            'epoch': epoch,
            'best_iou': best_iou, 
            'model_state_dict': model.state_dict(), 
            'optim_state_dict': optimizer.state_dict(),
            'logger_data': logger.data
        }
        save_checkpoint(state, is_best)

