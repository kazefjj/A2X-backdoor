import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
    parser.add_argument('--number_class', type=int, default=10, help='number of class')
    parser.add_argument('--model_name', type=str, default='resnet', help='name of model,mobilenet,resnet,vgg')    

    parser.add_argument('--seed', type=int, default=45, help='random seed')
  

    # backdoor attacks
    parser.add_argument('--poison_rate', type=float, default=0.005, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int, default=1, help='class of target label(only A2O)')
    parser.add_argument('--attack_type', type=str, default='A2X', help='type of backdoor label,A2O/A2A/A2A+/A2X/A2X+')
    parser.add_argument('--trigger_type', type=str, default='badnets', help='type of trigger')
    parser.add_argument('--attack_class_number', type=int, default=5, help='number of target label')
    parser.add_argument('--datasets_root_dir', type=str, default='./datasets', help='number of target label')
    parser.add_argument('--save_dir', type=str, default='./experiments', help='model save')

    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='SGD', help='select optimizer')
    parser.add_argument('--epoch', type=int, default=100, help='The size of epoch')
    parser.add_argument('--beign_train', type=bool, default=False, help='Train surrogate model')
    return parser.parse_args()