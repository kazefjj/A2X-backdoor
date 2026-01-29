# early_exit_experiments.py
# runs the experiments in section 5.1 

import torch
import numpy as np
import json
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import core
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, MNIST, CIFAR10
from torchvision.transforms import Compose
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, ToPILImage, Resize
from datasets import TinyImagenet
from sklearn import metrics
from core.utils import save
import os.path as osp
import config
opt = config.get_arguments()
number_class = opt.number_class
y_target = opt.target_label
poi_rate = opt.poison_rate
attack_type = opt.attack_type
attack_class_number = opt.attack_class_number
def visualize_tsne(list_data, labels, perplexity=30, n_iter=1000, title="t-SNE Visualization", num_classes=10):

    list_data = torch.cat(list_data, dim=0)
    list_data = [tensor.cpu().numpy() for tensor in list_data]
    

    sample_indices = range(len(list_data)) 
    print(len(list_data))
    
    data = np.array(list_data)
    data = data[sample_indices]
    labels = np.array(labels)


    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=n_iter)
    data_tsne = tsne.fit_transform(data)

    # 可视化
    plt.figure(figsize=(12, 10))
    for i in range(num_classes):

        class_indices = np.where(labels == i)[0]

        plt.scatter(data_tsne[class_indices, 0], data_tsne[class_indices, 1], label=f'Class {i}', s=15)


    plt.legend(loc='best', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.savefig(f"./outputs/t-sne_{title}_change.png")
    plt.clf()

def compute_class_mean(features, labels):
    unique_classes = np.unique(labels)
    class_means = {}
    for cls in unique_classes:
        class_means[cls] = features[labels == cls].mean(axis=0)
    return class_means
def compute_distance_matrix(class_means):
    classes = list(class_means.keys())
    num_classes = len(classes)
    
    distance_matrix = np.zeros((num_classes, num_classes))
    

    for i in range(num_classes):
        for j in range(i, num_classes): 
            if i == j:
                distance_matrix[i, j] = 0  
            else:
                dist = np.linalg.norm(class_means[classes[i]]-class_means[classes[j]],ord=2)
                #dist = np.linalg.norm(class_means[classes[i]]-class_means[classes[j]],ord=1)
                #dist = np.linalg.norm(class_means[classes[i]]-class_means[classes[j]],ord=np.inf)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist  
    
    return distance_matrix
def max_weight_matching(adjacency_matrix, num_edges=10):

    cost_matrix = -adjacency_matrix
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    selected_edges = []
    total_weight = 0
    
    for i in range(num_edges):
        row = row_ind[i]
        col = col_ind[i]
        weight = adjacency_matrix[row, col]
        selected_edges.append((row, col, weight))
        total_weight += weight
    
    return selected_edges, total_weight
def max_weight_matching_with_groups(adjacency_matrix, groups,num_edges):

    unique_groups = sorted(set(groups))  
    group_count = len(unique_groups)
    
    new_matrix = np.zeros((group_count, adjacency_matrix.shape[1]))
    for i, group in enumerate(unique_groups):

        group_indices = [index for index, value in enumerate(groups) if value == group]

        new_matrix[i] = np.sum(adjacency_matrix[group_indices, :], axis=0)
    

    cost_matrix = -new_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    selected_edges = []
    total_weight = 0
    for i in range(num_edges):
        row = row_ind[i]
        col = col_ind[i]
        
        indices = [index for index, value in enumerate(groups) if value == row]
        for indice in indices:
            weight = adjacency_matrix[indice, col]
            selected_edges.append([indice, col, weight])
        
            total_weight += weight
    return selected_edges,total_weight


def kmeans_clustering(data_dict, k=5):
    X = np.array(list(data_dict.values()))

    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)

    labels = kmeans.labels_

    centroids = kmeans.cluster_centers_

    return labels, centroids
def load_dict(model_path):
    print(torch.version.cuda)
    state_dict = torch.load(model_path,torch.device('cpu'))
    if 'model' in list(state_dict.keys()):
        return state_dict['model']
    else:
        return state_dict
def mapping(device='cpu'):
    features = []
    labels_list = []
    K = attack_class_number
    if opt.model_name == 'resnet':
        model=core.models.ResNet(18,number_class)
        last_name = 'linear'
    elif opt.model_name == 'vgg':
        model=core.models.vgg16_bn(number_class)
        last_name = 'classifier'
    elif opt.model_name == 'mobile':
        model=core.models.MobileNet()
        last_name = 'linear'
    dataset_name = opt.dataset
    model_path = osp.join(opt.save_dir, opt.model_name +'_'+ opt.dataset +'_'+ opt.trigger_type +'_'+ opt.attack_type + '_' + str(opt.attack_class_number) + '_'+ str(opt.poison_rate)+'/ckpt_epoch_100.pth')
    model.load_state_dict(load_dict(model_path))
    model.to(device)
    dataset = torchvision.datasets.CIFAR10


    transform_train = Compose([
        ToTensor(),
        RandomHorizontalFlip()
    ])
    trainset = dataset('./datasets', train=True, transform=transform_train, download=True)
    testset = dataset('./datasets', train=False, transform=transform_train, download=True)
    #trainset = TinyImagenet.TinyImageNet('./datasets' + '/tiny-imagenet-200', train=True,transform=transform_train)
    #testset = TinyImagenet.TinyImageNet('./datasets' + '/tiny-imagenet-200', train=False,transform=transform_train)
    image_size = 32
    pattern = torch.zeros((image_size,image_size), dtype=torch.uint8)
    pattern[-3:, -3:] = 255
    weight = torch.zeros((image_size, image_size), dtype=torch.float32)
    weight[-3:, -3:] = 1.0
    attack = core.BadNets(
        train_dataset=trainset,
        test_dataset=testset,
        model=None,
        loss=None,
        y_target=1,
        attack_type="no",
        attack_class=0,
        poisoned_rate=1,
        pattern=pattern,
        weight=weight,
        seed=0,
        deterministic=True
    )
    poisoned_trainset, poisoned_testset = attack.get_poisoned_dataset()
    train_loader = DataLoader(
                poisoned_trainset,
                batch_size=128,
                shuffle=True,
                num_workers=4,
                drop_last=False,
                pin_memory=True,
            )
    def hook(module, input, output):
        features.append(output.cpu().numpy())
    
    for name, module in model.named_modules():
        if name == last_name:#linear
            hook_handle = module.register_forward_hook(hook)
        
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            labels_list.append(b_y.cpu().numpy())
            output = model(b_x)        
    hook_handle.remove()
    all_labels = np.concatenate(labels_list, axis=0)
    all_features = np.concatenate(features, axis=0)
    class_means = compute_class_mean(all_features, all_labels)
    distance_matrix = compute_distance_matrix(class_means)

    pairs,_ = kmeans_clustering(class_means,K)

    max_mapping,_ = max_weight_matching_with_groups(distance_matrix,pairs,K)
    print(max_mapping)
    data_dict = {int(per_select[0]) : int(per_select[1]) for per_select in max_mapping}
    with open(f"experiments/Mappings/mapping_{dataset_name}_A2{K}.json", "w") as f:
        json.dump(data_dict, f, indent=4)
    with open(f"experiments/Mappings/mapping_{dataset_name}_A2{K}.json", "r") as f:
        mapping = json.load(f)
    mapping = {int(k): v for k, v in mapping.items()}
    print(mapping)

    #af.visualize_tsne(features, all_labels, perplexity=50, n_iter=3000, title="CIFAR-10 t-SNE Visualization", num_classes=10)
def get_pytorch_device():
    device = 'cpu'
    cuda = torch.cuda.is_available()
    print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)
    if cuda:
        device = 'cuda'
    return device

def main():

    device = get_pytorch_device()
    mapping(device)

if __name__ == '__main__':
    main()