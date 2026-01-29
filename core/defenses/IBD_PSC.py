# This is the test code of IBD-PSC defense.
# IBD-PSC: Input-level Backdoor Detection via Parameter-oriented Scaling Consistency [ICML, 2024] (https://arxiv.org/abs/2405.09786) 

import os
import pdb
import torch
from torchvision import transforms
from sklearn import metrics
from tqdm import tqdm
import copy
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import Subset
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from core.utils import save
from ..utils import test

from .base import Base


class IBD_PSC(Base):
    
    def __init__(self, model, n=5, xi=0.6, T = 0.9, scale=1.5, valset=None, seed=666, deterministic=False):
        super(IBD_PSC, self).__init__(seed, deterministic)
        self.model = model
        self.model.cuda()
        self.model.eval()
        self.n = n
        self.xi = xi
        self.T = T
        self.scale = scale
        self.valset = valset

        layer_num = self.count_BN_layers()
        sorted_indices = list(range(layer_num))
        sorted_indices = list(reversed(sorted_indices))
        self.sorted_indices = sorted_indices
        self.start_index = self.prob_start(self.scale, self.sorted_indices, valset=self.valset)

    
    def count_BN_layers(self):
        layer_num = 0
        for (name1, module1) in self.model.named_modules():
            if isinstance(module1, torch.nn.BatchNorm2d):
            # if isinstance(module1, torch.nn.Conv2d):
                layer_num += 1
        return layer_num
    
    # test accuracy on the dataset 
    def test_acc(self, dataset, schedule):
        """Test repaired curve model on dataset

        Args:
            dataset (types in support_list): Dataset.
            schedule (dict): Schedule for testing.
        """
        model = self.model
        result = test(model, dataset, schedule)
        return result
    def scale_var_index(self, index_bn, scale=1.5):
        copy_model = copy.deepcopy(self.model)
        index  = -1
        for (name1, module1) in copy_model.named_modules():
            if isinstance(module1, torch.nn.BatchNorm2d):
                index += 1
                if index in index_bn:
                    module1.weight.data *= scale
                    module1.bias.data *= scale

        return copy_model  
    def prob_start(self, scale, sorted_indices, valset):
        val_loader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False)
        layer_num = len(sorted_indices)
        # layer_index: k
        for layer_index in range(1, layer_num):            
            layers = sorted_indices[:layer_index]
            # print(layers)
            smodel = self.scale_var_index(layers, scale=scale)
            smodel.cuda()
            smodel.eval()
            
            total_num = 0 
            clean_wrong = 0
            with torch.no_grad():
                for idx, batch in enumerate(val_loader):
                    clean_img = batch[0]
                    labels = batch[1]
                    clean_img = clean_img.cuda()  # batch * channels * hight * width
                    # labels = labels.cuda()  # batch
                    clean_logits = smodel(clean_img).detach().cpu()
                    clean_pred = torch.argmax(clean_logits, dim=1)# model prediction
                    
                    clean_wrong += torch.sum(labels != clean_pred)
                    total_num += labels.shape[0]
                wrong_acc = clean_wrong / total_num
                # print(f'wrong_acc: {wrong_acc}')
                if wrong_acc > self.xi:
                    return layer_index

    def _test(self, dataset):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        self.model.eval()
        total_num = 0
        all_psc_score = []
        pred_correct_mask = []

        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                imgs = batch[0]
                labels = batch[1]
                total_num += labels.shape[0]
                imgs = imgs.cuda()  # batch * channels * hight * width
                labels = labels.cuda()  # batch
                original_pred = torch.argmax(self.model(imgs), dim=1) # model prediction
                mask = torch.eq(labels, original_pred) # only look at those samples that successfully attack the DNN
                pred_correct_mask.append(mask)

                psc_score = torch.zeros(labels.shape)
                scale_count = 0
                models = []
                for layer_index in range(self.start_index, self.start_index + self.n):
                    layers = self.sorted_indices[:layer_index+1]
                    # print(f'layers: {layers}')
                    smodel = self.scale_var_index(layers, scale=self.scale)
                    scale_count += 1
                    smodel.eval()
                    models.append(smodel)
                    logits = smodel(imgs).detach().cpu()
                    softmax_logits = torch.nn.functional.softmax(logits, dim=1)
                    psc_score += softmax_logits[torch.arange(softmax_logits.size(0)), original_pred.cpu()]

                psc_score /= scale_count
                all_psc_score.append(psc_score)
        
        all_psc_score = torch.cat(all_psc_score, dim=0).cpu()
        #pred_correct_mask = torch.cat(pred_correct_mask, dim=0).cpu()
        #all_psc_score = all_psc_score[pred_correct_mask]
        return all_psc_score
    def test(self, testset, poisoned_testset):
        print(f'start_index: {self.start_index}')

        benign_psc = self._test(testset)
        poison_psc = self._test(poisoned_testset)

        num_benign = benign_psc.size(0)
        num_poison = poison_psc.size(0)

        y_true = torch.cat((torch.zeros_like(benign_psc), torch.ones_like(poison_psc))).cpu().numpy()
        y_score = torch.cat((benign_psc, poison_psc), dim=0).cpu().numpy()
        y_pred = (y_score >= self.T)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        myf1 = metrics.f1_score(y_true, y_pred)
        print("TPR: {:.2f}".format(tp / (tp + fn) * 100))
        print("FPR: {:.2f}".format(fp / (tn + fp) * 100))
        print("AUC: {:.4f}".format(auc))
        save.log_IBD_to_csv(tp / (tp + fn) * 100,fp / (tn + fp) * 100,auc,'IBD.csv')
        print(f"f1 score: {myf1}")
    def split_trainset(self, trainset):
        print(f'start_index: {self.start_index}')

        trainset_psc = self._test(trainset)
        indices = torch.argsort(trainset_psc,descending=True)
        num_poisoned = int(0.01 * len(indices))
        indices = indices[:num_poisoned]
        #score_mask = trainset_psc >= self.T  
        #indices = torch.nonzero(score_mask).flatten()  
        print(len(indices))
        images = torch.stack([data[0] for data in trainset])
        labels = torch.LongTensor([data[1] for data in trainset])
        
        poisoned_images = images[indices]
        poisoned_labels = labels[indices]
        poisoned_dataset = TensorsDataset(poisoned_images, poisoned_labels)
        return poisoned_dataset
    def fine_tuning(self,trainset):
        device = torch.device("cuda:0")
        model = self.model.to(device)

        criterion = nn.CrossEntropyLoss()
        poi_set = self.split_trainset(trainset)
        
        train_loader = torch.utils.data.DataLoader(
            poi_set,
            batch_size=128,
            shuffle=True,
            num_workers=4,
            drop_last=False,
            pin_memory=True,
        )
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
        iteration = 0
        for i in range(5):
            for batch_id, batch in enumerate(train_loader):
                batch_img = batch[0]
                batch_label = batch[1]
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                optimizer.zero_grad()
                predict_digits = model(batch_img)
                loss = criterion(predict_digits, batch_label)
                (-loss).backward()
                optimizer.step()

                iteration += 1
        
        self.model = model
        model.eval()
    def _detect(self, inputs):
        inputs = inputs.cuda()
        self.model.eval()
        self.model.cuda()
        original_pred = torch.argmax(self.model(inputs), dim=1) # model prediction

        psc_score = torch.zeros(inputs.size(0))
        scale_count = 0
        for layer_index in range(self.start_index, self.start_index + self.n):
            layers = self.sorted_indices[:layer_index+1]
            # print(f'layers: {layers}')
            smodel = self.scale_var_index(layers, scale=self.scale)
            scale_count += 1
            smodel.eval()
            logits = smodel(inputs).detach().cpu()
            softmax_logits = torch.nn.functional.softmax(logits, dim=1)
            psc_score += softmax_logits[torch.arange(softmax_logits.size(0)), original_pred.cpu()]

        psc_score /= scale_count
        
        y_pred = psc_score >= self.T
        return y_pred
    
    def detect(self, dataset):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                imgs = batch[0]
                return self._detect(imgs)
class TensorsDataset(torch.utils.data.Dataset):


    def __init__(self, data_tensor, target_tensor=None, transforms=None, target_transforms=None):
        if target_tensor is not None:
            assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

        if transforms is None:
            transforms = []
        if target_transforms is None:
            target_transforms = []

        if not isinstance(transforms, list):
            transforms = [transforms]
        if not isinstance(target_transforms, list):
            target_transforms = [target_transforms]

        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):

        data_tensor = self.data_tensor[index]
        for transform in self.transforms:
            data_tensor = transform(data_tensor)

        if self.target_tensor is None:
            return data_tensor

        target_tensor = self.target_tensor[index]
        for transform in self.target_transforms:
            target_tensor = transform(target_tensor)

        return data_tensor, target_tensor

    def __len__(self):
        return self.data_tensor.size(0)