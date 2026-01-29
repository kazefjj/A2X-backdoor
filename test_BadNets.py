'''
This is the test code of poisoned training under BadNets.
'''


import os.path as osp
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize
import config
import core
from datasets import TinyImagenet

# ========== Set global settings ==========


CUDA_SELECTED_DEVICES = '0'

opt = config.get_arguments()
# CUDA_SELECTED_DEVICES = '0,3'
CUDA_SELECTED_DEVICES = '0'
deterministic = True
torch.manual_seed(opt.seed)
datasets_root_dir = opt.datasets_root_dir
number_class = opt.number_class
y_target = opt.target_label
poi_rate = opt.poison_rate
attack_type = opt.attack_type
attack_class_number = opt.attack_class_number
#model selction
if opt.model_name == 'resnet':
    poi_model=core.models.ResNet(18,number_class)
elif opt.model_name == 'vgg':
    poi_model=core.models.vgg16_bn(number_class)
elif opt.model_name == 'mobile':
    poi_model=core.models.MobileNet()


transform_test = Compose([
        ToTensor()
    ])
transform_train = Compose([
        RandomHorizontalFlip(),
        ToTensor()
    ])
#dataset selection
if opt.dataset == 'CIFAR10': 
    dataset = torchvision.datasets.CIFAR10
    trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
    testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
    image_size = 32
elif opt.dataset == 'CIFAR100': 
    dataset = torchvision.datasets.CIFAR100
    trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=True)
    testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=True)
    image_size = 32
elif opt.dataset == 'TinyImageNet': 
    trainset = TinyImagenet.TinyImageNet(datasets_root_dir + '/tiny-imagenet-200', train=True,transform=transform_train)
    testset = TinyImagenet.TinyImageNet(datasets_root_dir + '/tiny-imagenet-200', train=False,transform=transform_test)
    image_size = 64
#trigger selection
if opt.trigger_type == 'badnets':
    pattern = torch.zeros((image_size, image_size), dtype=torch.uint8)
    pattern[-3:, -3:] = 255
    weight = torch.zeros((image_size, image_size), dtype=torch.float32)
    weight[-3:, -3:] = 1.0
elif opt.trigger_type == 'badnets-random':
    pattern = torch.zeros((image_size, image_size), dtype=torch.uint8)
    pattern[-3:, -3:] = torch.tensor(np.random.randint(low=0, high=256, size=9).reshape(3, 3))
    weight = torch.zeros((image_size, image_size), dtype=torch.float32)
    weight[-3:, -3:] = 1.0
elif opt.trigger_type == 'blend':
    pattern = torch.zeros((image_size, image_size), dtype=torch.uint8)
    pattern[:, :] = torch.tensor(np.random.randint(low=0, high=256, size=image_size*image_size).reshape(image_size, image_size))
    weight = torch.zeros((image_size, image_size), dtype=torch.float32)
    weight[:, :] = 0.2
elif opt.trigger_type == 'lc':
    pattern = torch.zeros((image_size, image_size), dtype=torch.uint8)
    k = 6
    pattern[:k,:k] = 255
    pattern[:k,-k:] = 255
    pattern[-k:,:k] = 255
    pattern[-k:,-k:] = 255
    weight = torch.zeros((image_size, image_size), dtype=torch.float32)
    weight[:k,:k] = 1.0
    weight[:k,-k:] = 1.0
    weight[-k:,:k] = 1.0
    weight[-k:,-k:] = 1.0
elif opt.trigger_type == 'sig':
    overlay = np.zeros((image_size, image_size), np.float32)
    for i in range(image_size):
        overlay[:, i] = 20 * np.sin(2 * np.pi * i * 6 / image_size) + 20
    pattern = torch.from_numpy(overlay)
    weight = torch.zeros((image_size, image_size), dtype=torch.float32)
    weight[:, :] = 0.2
attack = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=poi_model,
    loss=nn.CrossEntropyLoss(),
    y_target=y_target,
    attack_type=attack_type,
    attack_class=attack_class_number,
    poisoned_rate=poi_rate,
    pattern=pattern,
    weight=weight,
    seed=opt.seed,
    deterministic=deterministic
)



schedule = {
    'device': 'GPU',
    'CUDA_SELECTED_DEVICES': CUDA_SELECTED_DEVICES,
    'GPU_num': 1,
    'benign_training': opt.beign_train,
    'batch_size': opt.batch_size,
    'num_workers': 4,

    'lr': opt.lr,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [50, 80],

    'epochs': 100,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': opt.save_dir,
    'experiment_name': opt.model_name +'_'+ opt.dataset +'_'+ opt.trigger_type +'_'+ opt.attack_type + '_' + str(opt.attack_class_number) + '_'+ str(opt.poison_rate), 
    'optimizer': opt.optimizer
}

attack.train(schedule)
attack.test(schedule)



