# Enhancing All-to-X Backdoor Attacks with Optimized Target Class Mapping

**Abstract**  *:Backdoor attacks pose severe threats to machine learning systems, prompting extensive research in this area. However, most existing work focuses on single-target All-to-One (A2O) attacks, overlooking the more complex All-to-X (A2X) attacks with multiple target classes, which are often assumed to have low attack success rates. In this paper, we first demonstrate that A2X attacks are robust against state-of-the-art defenses. We then propose a novel attack strategy that enhances the success rate of A2X attacks while maintaining stealthiness by optimizing grouping and backdoor mapping mechanisms. Our method improves the attack success rate by up to 28\% , with average improvements of 6.7\%, 16.4\%, 14.1\% on CIFAR10, CIFAR100, and Tiny-ImageNet, respectively. We anticipate that this study will raise awareness of A2X attacks and stimulate further research in this under-explored area.*

---

## Pre-requisites

Requirements:

+ python 3.9
+ pyTorch 2.4.0
+ CUDA 12.4
+ scikit-learn 1.6.0
+ scipy 1.13.1

Datasets：

- CIFAR-10/100 (downloaded automatically)
- [Tiny-imagenet](`http://cs231n.stanford.edu/tiny-imagenet-200.zip`)
- Download and place it under datasets/

---

## Surrogate Model Training(CIFAR10)

```
python test_BadNets.py --dataset CIFAR10 --number_class 10 --model_name resnet --beign_train True --optimizer SGD --lr 0.1 --attack_class_number 5 --save_dir ./experiments
```

---

## Mapping Selection(X=5)

```
python mapping_selection.py --dataset CIFAR10 --number_class 10 --model_name resnet  --trigger_type badnets --attack_class_number 5  --save_dir ./experiments
```

A JSON mapping file will be generated and stored in experiments/Mappings/

---

## Victim Model Training

The random mapping and optimized mapping correspond to A2X and A2X+, respectively.

```
python test_BadNets.py --dataset CIFAR10  --number_class 10 --model_name resnet --poison_rate  0.005 --attack_type A2X+ --trigger_type badnets  --attack_class_number 5
```

---