# Learning-By-Teaching
Learning Implicit Generative Models by Teaching Explicit Ones

## Important Dependencies
    - Python 2.7.14
    - tensorflow-gpu 1.12.0
    - numpy 1.15.4
    - scikit-learn 0.20.2
    - scipy 1.1.0

## Datasets
    - The default path for data is '/home/Data/[dataset_name]'. Please change it in dataset.py
    - The MNIST dataset will be downloaded automatically.
    - Cifar10 can be downloaded from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    - CelebA dataset is cropped into 64x64

## Toy Data:
    - ring (LBT): python LBT_toy.py -dataset ring -mog_scale 1.0 -mog_std 0.1 -gpu [GPU_ID]
    - ring (LBT-GAN): python LBT-GAN_toy.py -dataset ring -mog_scale 1.0 -mog_std 0.1 -gpu [GPU_ID]

    - grid (LBT): python LBT_toy.py -dataset grid -n_mixture 100 -mog_scale 0.2 -mog_std 0.01 -batch_size 2048 -batch_size_est 2048 -max_iter 2000000 -n_est 3 -n_viz 51200 -gpu [GPU_ID]
    - grid (LBT-GAN): python LBT-GAN_toy.py -dataset grid -n_mixture 100 -mog_scale 0.2 -mog_std 0.01 -batch_size 2048 -batch_size_est 2048 -max_iter 2000000 -n_est 3 -n_viz 51200 -gpu [GPU_ID]

## Stacked-MNIST:
    - Baseline: python LBT-GAN_smnist.py -gpu [GPU_ID]
    - LBT-GAN: python LBT-GAN_smnist.py -lbt -gpu [GPU_ID]

## Cifar10 & MNIST:
    - Cifar10: python LBT-GAN_cifar10.py -lbt -gpu [GPU_ID]
    - CelebA: python LBT-GAN_celeba.py -lbt -gpu [GPU_ID]

